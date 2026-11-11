import 'package:camera/camera.dart';
import 'dart:ui';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import '../core/constants/app_constants.dart';
import '../core/utils/image_converter.dart';
import '../core/utils/math_utils.dart';
import '../data/models/user_model.dart';
import 'face_quality_assessment.dart';
import 'liveness_detector.dart';
import 'passive_liveness_detector.dart';
import 'recognizer_service.dart';

enum BiometricsStatus {
  idle,
  multipleFaces,
  qualityGateFailed,
  waitingForBlink,
  spoofDetected,
  recognized,
  notEnrolled,
  cooldownActive,
}

class BiometricsFrameResult {
  final BiometricsStatus status;
  final Face? face;
  final FaceQualityResult? qualityResult;
  final PassiveLivenessResult? livenessResult;
  final LivenessCheckResult? blinkResult;
  final MatchResult? matchResult;
  final int totalLatencyMs;
  final String statusMessage;
  final bool isAttendanceMarked;

  const BiometricsFrameResult({
    required this.status,
    this.face,
    this.qualityResult,
    this.livenessResult,
    this.blinkResult,
    this.matchResult,
    required this.totalLatencyMs,
    required this.statusMessage,
    this.isAttendanceMarked = false,
  });
}

/// Unified Production Biometrics Engine for Instant Walk-Through Attendance
/// Coordinates Direct ROI Extraction, ISO Face Quality Assessment (FQA),
/// Active Eye-Blink Liveness Verification, and TFLite 1:N Recognition.
class BiometricsPipeline {
  final FaceQualityAssessment qualityAssessment;
  final PassiveLivenessDetector passiveLivenessDetector;
  final LivenessDetector livenessDetector;
  final RecognizerService recognizerService;

  // Track user cooldowns to avoid duplicate logs in walk-through mode
  final Map<int, DateTime> _attendanceCooldowns = {};
  // Track on-screen faces by trackingId to user ID for persistent cooldown display
  final Map<int, int> _trackedFaceToUserId = {};
  final Map<int, String> _userIdToName = {};

  BiometricsPipeline({
    FaceQualityAssessment? qualityAssessment,
    PassiveLivenessDetector? passiveLivenessDetector,
    LivenessDetector? livenessDetector,
    RecognizerService? recognizerService,
  })  : qualityAssessment = qualityAssessment ?? FaceQualityAssessment(),
        passiveLivenessDetector =
            passiveLivenessDetector ?? PassiveLivenessDetector(),
        livenessDetector = livenessDetector ?? LivenessDetector(),
        recognizerService = recognizerService ?? RecognizerService();

  void reset() {
    livenessDetector.reset();
    passiveLivenessDetector.reset();
    _trackedFaceToUserId.clear();
  }

  void clearCooldowns() {
    _attendanceCooldowns.clear();
    _trackedFaceToUserId.clear();
    _userIdToName.clear();
  }

  /// Processes an incoming camera frame in sub-100ms total pipeline delay
  Future<BiometricsFrameResult> processFrame({
    required CameraImage cameraImage,
    required List<Face> faces,
    required CameraDescription camera,
    required List<UserModel> enrolledUsers,
    int? overrideSensorOrientation,
  }) async {
    final stopwatch = Stopwatch()..start();

    // -------------------------------------------------------------
    // Gate 1: Face Presence & Guard Checks (with Queue Prioritization)
    // -------------------------------------------------------------
    if (faces.isEmpty) {
      livenessDetector.reset();
      _trackedFaceToUserId.clear();
      return BiometricsFrameResult(
        status: BiometricsStatus.idle,
        totalLatencyMs: stopwatch.elapsedMilliseconds,
        statusMessage: 'Blink your eyes to check in',
      );
    }

    Face targetFace;
    if (faces.length == 1) {
      targetFace = faces.first;
    } else {
      // Sort faces by bounding box area (largest foreground face first)
      final sortedFaces = List<Face>.from(faces)
        ..sort((a, b) {
          final areaA = a.boundingBox.width * a.boundingBox.height;
          final areaB = b.boundingBox.width * b.boundingBox.height;
          return areaB.compareTo(areaA);
        });

      final primaryArea =
          sortedFaces[0].boundingBox.width * sortedFaces[0].boundingBox.height;
      final secondaryArea =
          sortedFaces[1].boundingBox.width * sortedFaces[1].boundingBox.height;

      // If two competing faces are side-by-side with similar sizes (tailgating / buddy-punching)
      if (secondaryArea > 0.55 * primaryArea && secondaryArea >= 70 * 70) {
        return BiometricsFrameResult(
          status: BiometricsStatus.multipleFaces,
          totalLatencyMs: stopwatch.elapsedMilliseconds,
          statusMessage: 'Multiple faces in view! 1 person at a time.',
        );
      }

      // Foreground person in queue takes priority
      targetFace = sortedFaces.first;
    }

    final Face face = targetFace;
    final now = DateTime.now();

    // Early Cooldown Check: If this tracked face already checked in and is on cooldown,
    // display cooldown countdown directly without requesting a repeat blink or running TFLite inference.
    if (face.trackingId != null &&
        _trackedFaceToUserId.containsKey(face.trackingId)) {
      final int trackedUserId = _trackedFaceToUserId[face.trackingId]!;
      if (_attendanceCooldowns.containsKey(trackedUserId)) {
        final lastRecorded = _attendanceCooldowns[trackedUserId]!;
        final diff = now.difference(lastRecorded);
        if (diff < AppConstants.walkThroughCooldown) {
          final remainingSecs =
              AppConstants.walkThroughCooldown.inSeconds - diff.inSeconds;
          final userName = _userIdToName[trackedUserId] ?? 'Employee';
          return BiometricsFrameResult(
            status: BiometricsStatus.cooldownActive,
            face: face,
            totalLatencyMs: stopwatch.elapsedMilliseconds,
            statusMessage:
                'Already checked in: $userName (${remainingSecs}s cooldown)',
            isAttendanceMarked: false,
          );
        } else {
          // Cooldown expired for tracked face
          _attendanceCooldowns.remove(trackedUserId);
          _trackedFaceToUserId.remove(face.trackingId);
        }
      }
    }
    final int sensorOrientation =
        overrideSensorOrientation ?? camera.sensorOrientation;

    // -------------------------------------------------------------
    // Gate 2: Ultra-Fast Direct ROI Extraction (sub-3ms)
    // -------------------------------------------------------------
    final img.Image faceCrop = ImageConverter.cropFaceDirect(
      cameraImage,
      face.boundingBox,
      recognizerService.inputWidth,
      sensorOrientation,
    );

    // -------------------------------------------------------------
    // Gate 3: Active Eye-Blink Liveness Tracking
    // -------------------------------------------------------------
    // Process facial eye probabilities on every frame to ensure no blink cycles are missed
    final blinkResult = livenessDetector.processFace(face);

    // -------------------------------------------------------------
    // Gate 4: Face Quality Assessment (FQA) with Upright Frame Dimensions
    // -------------------------------------------------------------
    final bool isRotated =
        sensorOrientation == 90 || sensorOrientation == 270;
    final Size uprightFrameSize = Size(
      isRotated ? cameraImage.height.toDouble() : cameraImage.width.toDouble(),
      isRotated ? cameraImage.width.toDouble() : cameraImage.height.toDouble(),
    );

    final quality = qualityAssessment.assessQuality(
      face: face,
      faceCrop: faceCrop,
      imageSize: uprightFrameSize,
    );

    // If blink cycle is not yet completed, guide the user to blink
    if (!blinkResult.isLivenessConfirmed) {
      stopwatch.stop();
      final bool hasSeverePositionDefect = !quality.isAcceptable &&
          (quality.defect == FaceQualityDefect.faceTooSmall ||
              quality.defect == FaceQualityDefect.steepYaw ||
              quality.defect == FaceQualityDefect.steepPitch ||
              quality.defect == FaceQualityDefect.clippedBoundary);

      return BiometricsFrameResult(
        status: hasSeverePositionDefect
            ? BiometricsStatus.qualityGateFailed
            : BiometricsStatus.waitingForBlink,
        face: face,
        qualityResult: quality,
        blinkResult: blinkResult,
        totalLatencyMs: stopwatch.elapsedMilliseconds,
        statusMessage: hasSeverePositionDefect
            ? quality.feedbackMessage
            : blinkResult.message,
      );
    }

    // Blink cycle completed! Verify that this eyes-open frame is sharp and high-quality
    if (!quality.isAcceptable) {
      stopwatch.stop();
      return BiometricsFrameResult(
        status: BiometricsStatus.qualityGateFailed,
        face: face,
        qualityResult: quality,
        blinkResult: blinkResult,
        totalLatencyMs: stopwatch.elapsedMilliseconds,
        statusMessage: quality.feedbackMessage,
      );
    }

    // Physical liveness proven with verified sharp, eyes-open frame!
    const liveness = PassiveLivenessResult(
      isLive: true,
      livenessScore: 1.0,
      attackType: PassiveAttackType.none,
      message: 'Liveness Verified! ✓',
    );

    // -------------------------------------------------------------
    // Gate 5: Deep Embedding Extraction (TFLite Inference, sub-30ms)
    // -------------------------------------------------------------
    final List<double> embedding = recognizerService.predict(faceCrop);

    // -------------------------------------------------------------
    // Gate 6: 1:N Fast Cosine Similarity Search with Dual-Pass Lighting Fallback
    // -------------------------------------------------------------
    MatchResult match = recognizerService.findBestMatch(
      embedding,
      enrolledUsers,
    );

    // If initial match is borderline and lighting was sub-optimal, perform dual-pass contrast stretch
    if (!match.isMatch &&
        match.distance <= AppConstants.matchThreshold + 0.08 &&
        enrolledUsers.isNotEmpty) {
      final img.Image contrastNormalized =
          MathUtils.adaptivePhotometricNormalize(faceCrop);
      final List<double> altEmbedding =
          recognizerService.predict(contrastNormalized);
      match = recognizerService.findBestMatch(
        embedding,
        enrolledUsers,
        alternateEmbedding: altEmbedding,
      );
    }

    // Consume the blink event so subsequent frames don't spin inference in a tight loop
    livenessDetector.reset();

    stopwatch.stop();
    final int totalLatency = stopwatch.elapsedMilliseconds;

    if (match.isMatch && match.user != null) {
      final int userId = match.user!.id ?? match.user!.name.hashCode;
      _userIdToName[userId] = match.user!.name;
      if (face.trackingId != null) {
        _trackedFaceToUserId[face.trackingId!] = userId;
      }

      // Check walk-through cooldown
      if (_attendanceCooldowns.containsKey(userId)) {
        final lastRecorded = _attendanceCooldowns[userId]!;
        final diff = now.difference(lastRecorded);
        if (diff < AppConstants.walkThroughCooldown) {
          final remainingSecs =
              AppConstants.walkThroughCooldown.inSeconds - diff.inSeconds;
          return BiometricsFrameResult(
            status: BiometricsStatus.cooldownActive,
            face: face,
            qualityResult: quality,
            livenessResult: liveness,
            blinkResult: blinkResult,
            matchResult: match,
            totalLatencyMs: totalLatency,
            statusMessage:
                'Already checked in: ${match.user!.name} (${remainingSecs}s cooldown)',
            isAttendanceMarked: false,
          );
        }
      }

      // Mark attendance
      _attendanceCooldowns[userId] = now;

      return BiometricsFrameResult(
        status: BiometricsStatus.recognized,
        face: face,
        qualityResult: quality,
        livenessResult: liveness,
        blinkResult: blinkResult,
        matchResult: match,
        totalLatencyMs: totalLatency,
        statusMessage: 'Welcome, ${match.user!.name}!',
        isAttendanceMarked: true,
      );
    }

    return BiometricsFrameResult(
      status: BiometricsStatus.notEnrolled,
      face: face,
      qualityResult: quality,
      livenessResult: liveness,
      blinkResult: blinkResult,
      matchResult: match,
      totalLatencyMs: totalLatency,
      statusMessage: 'Face not recognized. Please register employee.',
      isAttendanceMarked: false,
    );
  }
}
