import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import '../core/constants/app_constants.dart';

enum BlinkPhase {
  waitingForOpen,
  eyesOpen,
  eyesClosed,
  blinkCompleted,
}

class LivenessCheckResult {
  final bool isLivenessConfirmed;
  final bool isStaticSpoofSuspected;
  final double progress;
  final String message;
  final BlinkPhase phase;

  const LivenessCheckResult({
    required this.isLivenessConfirmed,
    required this.isStaticSpoofSuspected,
    required this.progress,
    required this.message,
    required this.phase,
  });
}

class LivenessDetector {
  BlinkPhase _phase = BlinkPhase.waitingForOpen;
  DateTime? _closedPhaseStartTime;
  DateTime? _completedPhaseStartTime;
  int? _lastTrackingId;

  BlinkPhase get currentPhase => _phase;

  void reset() {
    _phase = BlinkPhase.waitingForOpen;
    _closedPhaseStartTime = null;
    _completedPhaseStartTime = null;
    _lastTrackingId = null;
  }

  /// Evaluates facial eye open probabilities from ML Kit and determines liveness.
  LivenessCheckResult processFace(Face face) {
    if (face.trackingId != null) {
      if (_lastTrackingId != null && _lastTrackingId != face.trackingId) {
        _phase = BlinkPhase.waitingForOpen;
        _closedPhaseStartTime = null;
        _completedPhaseStartTime = null;
      }
      _lastTrackingId = face.trackingId;
    }

    final double? leftOpen = face.leftEyeOpenProbability;
    final double? rightOpen = face.rightEyeOpenProbability;

    // If device or ML Kit hasn't returned probabilities yet
    if (leftOpen == null || rightOpen == null) {
      return const LivenessCheckResult(
        isLivenessConfirmed: false,
        isStaticSpoofSuspected: false,
        progress: 0.1,
        message: 'Blink your eyes to check in',
        phase: BlinkPhase.waitingForOpen,
      );
    }

    final double avgEyeOpen = (leftOpen + rightOpen) / 2.0;

    switch (_phase) {
      case BlinkPhase.waitingForOpen:
        // Both eyes must be confirmed open (prevents initial asymmetric winks)
        final bool isBothEyesOpen = leftOpen >= 0.50 &&
            rightOpen >= 0.50 &&
            (avgEyeOpen >= AppConstants.eyeOpenThreshold ||
                (leftOpen >= 0.60 && rightOpen >= 0.60));
        if (isBothEyesOpen) {
          _phase = BlinkPhase.eyesOpen;
          return const LivenessCheckResult(
            isLivenessConfirmed: false,
            isStaticSpoofSuspected: false,
            progress: 0.35,
            message: 'Blink your eyes to check in',
            phase: BlinkPhase.eyesOpen,
          );
        }
        return const LivenessCheckResult(
          isLivenessConfirmed: false,
          isStaticSpoofSuspected: false,
          progress: 0.15,
          message: 'Blink your eyes to check in',
          phase: BlinkPhase.waitingForOpen,
        );

      case BlinkPhase.eyesOpen:
        // Transition to eyes closed during natural blink
        // Both eyes must close during a natural blink (prevents winks or asymmetry)
        final bool isBothEyesClosed =
            (leftOpen <= 0.30 && rightOpen <= 0.30) ||
            (avgEyeOpen <= AppConstants.eyeClosedThreshold &&
                leftOpen <= 0.40 &&
                rightOpen <= 0.40);
        if (isBothEyesClosed) {
          _phase = BlinkPhase.eyesClosed;
          _closedPhaseStartTime = DateTime.now();
          return const LivenessCheckResult(
            isLivenessConfirmed: false,
            isStaticSpoofSuspected: false,
            progress: 0.70,
            message: 'Blink registered! Open eyes...',
            phase: BlinkPhase.eyesClosed,
          );
        }
        return const LivenessCheckResult(
          isLivenessConfirmed: false,
          isStaticSpoofSuspected: false,
          progress: 0.35,
          message: 'Blink your eyes to check in',
          phase: BlinkPhase.eyesOpen,
        );

      case BlinkPhase.eyesClosed:
        // Transition back to eyes open -> Complete blink cycle!
        // Both eyes must reopen to prevent winks from being captured for recognition
        final bool isBothEyesReopened = leftOpen >= 0.50 &&
            rightOpen >= 0.50 &&
            (avgEyeOpen >= AppConstants.eyeOpenThreshold ||
                (leftOpen >= 0.60 && rightOpen >= 0.60));
        if (isBothEyesReopened) {
          _phase = BlinkPhase.blinkCompleted;
          _completedPhaseStartTime = DateTime.now();
          return const LivenessCheckResult(
            isLivenessConfirmed: true,
            isStaticSpoofSuspected: false,
            progress: 1.0,
            message: 'Liveness Verified! ✓',
            phase: BlinkPhase.blinkCompleted,
          );
        }

        // If eyes remain closed for more than 2 seconds, reset
        if (_closedPhaseStartTime != null &&
            DateTime.now().difference(_closedPhaseStartTime!) >
                const Duration(milliseconds: 2000)) {
          _phase = BlinkPhase.waitingForOpen;
          _closedPhaseStartTime = null;
          return const LivenessCheckResult(
            isLivenessConfirmed: false,
            isStaticSpoofSuspected: false,
            progress: 0.20,
            message: 'Blink your eyes to check in',
            phase: BlinkPhase.waitingForOpen,
          );
        }

        return const LivenessCheckResult(
          isLivenessConfirmed: false,
          isStaticSpoofSuspected: false,
          progress: 0.70,
          message: 'Blink registered! Open eyes...',
          phase: BlinkPhase.eyesClosed,
        );

      case BlinkPhase.blinkCompleted:
        // If liveness confirmed frame is not consumed within 1500ms, reset
        if (_completedPhaseStartTime != null &&
            DateTime.now().difference(_completedPhaseStartTime!) >
                const Duration(milliseconds: 1500)) {
          _phase = BlinkPhase.waitingForOpen;
          _completedPhaseStartTime = null;
          return const LivenessCheckResult(
            isLivenessConfirmed: false,
            isStaticSpoofSuspected: false,
            progress: 0.35,
            message: 'Blink your eyes to check in',
            phase: BlinkPhase.waitingForOpen,
          );
        }
        return const LivenessCheckResult(
          isLivenessConfirmed: true,
          isStaticSpoofSuspected: false,
          progress: 1.0,
          message: 'Liveness Verified! ✓',
          phase: BlinkPhase.blinkCompleted,
        );
    }
  }
}
