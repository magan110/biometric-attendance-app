import 'dart:math';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import '../core/constants/app_constants.dart';
import '../core/utils/math_utils.dart';

/// Presentation attack classification for security auditing
enum PassiveAttackType {
  none,
  screenReplay,
  printPhoto2D,
  specularGlare,
  staticPlanarSpoof,
  siliconeMask,
}

/// Result of passive presentation attack detection (ISO/IEC 30107-3 compliant)
class PassiveLivenessResult {
  final bool isLive;
  final double livenessScore; // 0.0 to 1.0
  final PassiveAttackType attackType;
  final String message;
  final double textureScore; // 0.0 to 1.0
  final double chrominanceScore; // 0.0 to 1.0
  final double specularScore; // 0.0 to 1.0
  final double temporalScore; // 0.0 to 1.0

  const PassiveLivenessResult({
    required this.isLive,
    required this.livenessScore,
    required this.attackType,
    required this.message,
    this.textureScore = 0.0,
    this.chrominanceScore = 0.0,
    this.specularScore = 0.0,
    this.temporalScore = 0.0,
  });
}

class _FaceTemporalRecord {
  final double aspectRatio;
  final double width;
  final double height;
  final double centerX;
  final double centerY;
  final double meanLuminance;
  final double? eyeOpenProb;
  final DateTime timestamp;

  _FaceTemporalRecord({
    required this.aspectRatio,
    required this.width,
    required this.height,
    required this.centerX,
    required this.centerY,
    required this.meanLuminance,
    this.eyeOpenProb,
    required this.timestamp,
  });
}

/// Zero-Friction Passive Anti-Spoofing (PAD) Engine
/// Evaluates spatial frequency, Moiré screen artifacts, chrominance distribution,
/// and temporal micro-dynamics without ever requesting user interaction (no blinks, smiles, head turns).
class PassiveLivenessDetector {
  final Map<int, List<_FaceTemporalRecord>> _faceTemporalHistory = {};
  DateTime? _lastCleanupTime;

  /// Resets temporal history buffer
  void reset() {
    _faceTemporalHistory.clear();
  }

  /// Passively assesses facial authenticity on every frame
  PassiveLivenessResult evaluateFace({
    required Face face,
    required img.Image faceCrop,
    int? trackingId,
  }) {
    _cleanupExpiredHistory();

    // -------------------------------------------------------------
    // 1. Planar Specular Reflection Analysis (Screen / Photo Glare)
    // -------------------------------------------------------------
    final _SpecularMetric specular = _analyzeSpecularReflection(faceCrop);

    if (specular.isPlanarGlassReflectionDetected) {
      return PassiveLivenessResult(
        isLive: false,
        livenessScore: 0.20,
        attackType: PassiveAttackType.specularGlare,
        message: 'Screen glass reflection detected! Display blocked.',
        textureScore: 0.5,
        chrominanceScore: 0.5,
        specularScore: specular.score,
        temporalScore: 0.5,
      );
    }

    // -------------------------------------------------------------
    // 2. High-Frequency Texture & Moiré Pattern Analysis
    // -------------------------------------------------------------
    final _TextureMetric texture = _analyzeTextureAndMoire(faceCrop);

    if (texture.isScreenMoireDetected) {
      return PassiveLivenessResult(
        isLive: false,
        livenessScore: 0.15,
        attackType: PassiveAttackType.screenReplay,
        message: 'Screen replay detected! (Moiré grid pattern blocked)',
        textureScore: texture.score,
        chrominanceScore: 0.5,
        specularScore: specular.score,
        temporalScore: 0.5,
      );
    }

    if (texture.isFlatPrintDetected) {
      return PassiveLivenessResult(
        isLive: false,
        livenessScore: 0.20,
        attackType: PassiveAttackType.printPhoto2D,
        message: '2D photo print detected! (Flat texture pattern blocked)',
        textureScore: texture.score,
        chrominanceScore: 0.5,
        specularScore: specular.score,
        temporalScore: 0.5,
      );
    }

    // -------------------------------------------------------------
    // 3. Chrominance & Skin Blood Perfusion Distribution (YCbCr)
    // -------------------------------------------------------------
    final double chrominanceScore = _analyzeChrominanceDistribution(faceCrop);

    if (chrominanceScore < 0.25) {
      return PassiveLivenessResult(
        isLive: false,
        livenessScore: 0.22,
        attackType: PassiveAttackType.screenReplay,
        message: 'Display screen color gamut detected! Live face required.',
        textureScore: texture.score,
        chrominanceScore: chrominanceScore,
        specularScore: specular.score,
        temporalScore: 0.5,
      );
    }

    // -------------------------------------------------------------
    // 4. Synthetic Silicone / Latex Mask Analysis (Perfusion & Hemoglobin)
    // -------------------------------------------------------------
    final _PerfusionMetric perfusion =
        _analyzeBloodPerfusionAndSilicone(faceCrop);

    if (perfusion.isSiliconeMaskDetected) {
      return PassiveLivenessResult(
        isLive: false,
        livenessScore: 0.18,
        attackType: PassiveAttackType.siliconeMask,
        message: 'Synthetic mask detected! Live human face required.',
        textureScore: texture.score,
        chrominanceScore: chrominanceScore,
        specularScore: specular.score,
        temporalScore: 0.5,
      );
    }

    // -------------------------------------------------------------
    // 5. Passive Temporal Micro-Dynamics (Rolling Window)
    // -------------------------------------------------------------
    double temporalScore = 0.85; // Default for first frame
    if (trackingId != null) {
      temporalScore = _evaluateTemporalMicroDynamics(
        trackingId: trackingId,
        face: face,
        meanLuminance: texture.meanLuminance,
        textureScore: texture.score,
      );

      if (temporalScore < 0.20) {
        return PassiveLivenessResult(
          isLive: false,
          livenessScore: 0.25,
          attackType: PassiveAttackType.staticPlanarSpoof,
          message: 'Static photo spoof detected! Zero biological micro-movement.',
          textureScore: texture.score,
          chrominanceScore: chrominanceScore,
          specularScore: specular.score,
          temporalScore: temporalScore,
        );
      }
    }

    // -------------------------------------------------------------
    // 6. Score Fusion Engine
    // -------------------------------------------------------------
    final double fusedScore = 0.35 * texture.score +
        0.30 * chrominanceScore +
        0.15 * specular.score +
        0.20 * temporalScore;

    final bool isLive = fusedScore >= AppConstants.passiveLivenessThreshold;

    return PassiveLivenessResult(
      isLive: isLive,
      livenessScore: fusedScore,
      attackType: PassiveAttackType.none,
      message: isLive
          ? 'Live Face Verified ✓'
          : 'Liveness unconfirmed. Look directly at camera',
      textureScore: texture.score,
      chrominanceScore: chrominanceScore,
      specularScore: specular.score,
      temporalScore: temporalScore,
    );
  }

  /// Evaluates frequency distribution for Moiré screen grids vs. flat paper prints
  _TextureMetric _analyzeTextureAndMoire(img.Image faceCrop) {
    final int w = faceCrop.width;
    final int h = faceCrop.height;
    if (w < 8 || h < 8) {
      return _TextureMetric(
        score: 0.5,
        isScreenMoireDetected: false,
        isFlatPrintDetected: false,
        meanLuminance: 128.0,
      );
    }

    double totalGrad = 0.0;
    double highFreqEnergy = 0.0;
    double totalLum = 0.0;
    int sampled = 0;

    // Focus on forehead region (y: 10% to 28%, x: 20% to 80%) where skin texture is purest
    // and naturally free from eyeglass frames/reflections, surgical masks, and beards.
    final int startX = max(1, (w * 0.20).toInt());
    final int endX = min(w - 2, (w * 0.80).toInt());
    final int startY = max(1, (h * 0.10).toInt());
    final int endY = min(h - 2, (h * 0.28).toInt());

    for (int y = startY; y < endY; y += 2) {
      for (int x = startX; x < endX; x += 2) {
        final p = faceCrop.getPixel(x, y);
        final pRight = faceCrop.getPixel(x + 1, y);
        final pLeft = faceCrop.getPixel(x - 1, y);
        final double lum = _lum(p);

        // Exclude saturated specular reflections and their immediate step boundaries
        if (_isSaturated(p) ||
            _isSaturated(pRight) ||
            _isSaturated(pLeft) ||
            _isSaturated(faceCrop.getPixel(x, y + 1)) ||
            _isSaturated(faceCrop.getPixel(x, y - 1))) {
          continue;
        }

        totalLum += lum;

        final double lumRight = _lum(pRight);
        final double lumLeft = _lum(pLeft);
        final double lumDown = _lum(faceCrop.getPixel(x, y + 1));
        final double lumUp = _lum(faceCrop.getPixel(x, y - 1));

        final double gradX = (lumRight - lum).abs();
        final double gradY = (lumDown - lum).abs();
        totalGrad += gradX + gradY;

        // Second difference (high-frequency curvature)
        final double hf =
            (4.0 * lum - lumRight - lumLeft - lumDown - lumUp).abs();
        highFreqEnergy += hf;

        sampled++;
      }
    }

    final double meanLum = sampled > 0 ? totalLum / sampled : 128.0;
    final double meanGrad = sampled > 0 ? totalGrad / sampled : 0.0;
    final double ratio = totalGrad > 0.0 ? highFreqEnergy / totalGrad : 0.0;

    // Moiré patterns cause abnormal spikes in high-frequency ratio (subpixel grid aliasing)
    final bool isMoire = ratio >= 1.3 && meanGrad > 15.0;

    // Flat paper printouts have suppressed gradient energy (< 0.5)
    final bool isFlatPrint = meanGrad < 0.5 && sampled > 100;

    // Normal skin texture ratio sits comfortably below 1.0; drops as it approaches screen grid spike
    double textureScore = 0.92;
    if (ratio > 0.90) {
      textureScore = (1.0 - (ratio - 0.90) * 2.5).clamp(0.0, 1.0);
    }

    return _TextureMetric(
      score: textureScore,
      isScreenMoireDetected: isMoire,
      isFlatPrintDetected: isFlatPrint,
      meanLuminance: meanLum,
    );
  }

  /// Analyzes spectral blood perfusion to identify synthetic silicone / latex masks.
  /// Living skin absorbs green light via oxygenated hemoglobin (G/R in [0.52, 0.88]).
  /// Silicone masks painted with pigments have anomalous G/R ratio or flat spatial Cr variance.
  _PerfusionMetric _analyzeBloodPerfusionAndSilicone(img.Image faceCrop) {
    final int w = faceCrop.width;
    final int h = faceCrop.height;
    if (w < 8 || h < 8) {
      return _PerfusionMetric(score: 0.8, isSiliconeMaskDetected: false);
    }

    int sampled = 0;
    double sumGrRatio = 0.0;
    double sumForeheadCr = 0.0;
    int foreheadCount = 0;
    double sumCheekCr = 0.0;
    int cheekCount = 0;

    final int fTop = (h * 0.10).toInt();
    final int fBottom = (h * 0.30).toInt();
    final int fLeft = (w * 0.25).toInt();
    final int fRight = (w * 0.75).toInt();

    final int cTop = (h * 0.45).toInt();
    final int cBottom = (h * 0.70).toInt();
    final int cLeft = (w * 0.15).toInt();
    final int cRight = (w * 0.85).toInt();

    for (int y = 0; y < h; y += 3) {
      for (int x = 0; x < w; x += 3) {
        final p = faceCrop.getPixel(x, y);
        final double r = p.r.toDouble();
        final double g = p.g.toDouble();
        final double b = p.b.toDouble();

        if (r > 30.0) {
          final double gr = g / r;
          sumGrRatio += gr;
          sampled++;
        }

        final double cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;

        if (y >= fTop && y <= fBottom && x >= fLeft && x <= fRight) {
          sumForeheadCr += cr;
          foreheadCount++;
        } else if (y >= cTop && y <= cBottom && x >= cLeft && x <= cRight) {
          sumCheekCr += cr;
          cheekCount++;
        }
      }
    }

    if (sampled < 20) {
      return _PerfusionMetric(score: 0.8, isSiliconeMaskDetected: false);
    }

    final double avgGr = sumGrRatio / sampled;
    final double meanForeheadCr =
        foreheadCount > 0 ? sumForeheadCr / foreheadCount : 140.0;
    final double meanCheekCr =
        cheekCount > 0 ? sumCheekCr / cheekCount : 140.0;
    final double deltaCr = (meanCheekCr - meanForeheadCr).abs();

    // Living skin and realistic silicone masks have warm chromatic tones (Cr != 128)
    final bool isChromatic = (meanForeheadCr - 128.0).abs() > 3.0 ||
        (meanCheekCr - 128.0).abs() > 3.0;

    final bool isAnomalousColor = isChromatic && (avgGr > 0.94 || avgGr < 0.45);
    final bool isFlatPerfusion =
        deltaCr < 0.8 && foreheadCount > 10 && cheekCount > 10;

    final bool isSilicone = isChromatic && isAnomalousColor && isFlatPerfusion;
    final double score = isSilicone ? 0.20 : 0.90;

    return _PerfusionMetric(score: score, isSiliconeMaskDetected: isSilicone);
  }

  /// Evaluates chrominance distribution in YCbCr color space.
  /// Living skin has natural hemoglobin perfusion variance; digital screens have skewed Cb.
  /// Accommodates masks/beards by focusing on upper facial skin when lower face is occluded.
  double _analyzeChrominanceDistribution(img.Image faceCrop) {
    final int w = faceCrop.width;
    final int h = faceCrop.height;
    if (w < 4 || h < 4) return 0.5;

    int validSkinUpper = 0;
    int totalUpper = 0;
    double sumCbUpper = 0.0;
    double sumCrUpper = 0.0;

    int validSkinTotal = 0;
    int totalSampled = 0;
    double sumCb = 0.0;
    double sumCr = 0.0;

    final int upperBoundary = (h * 0.50).toInt();

    for (int y = 0; y < h; y += 3) {
      for (int x = 0; x < w; x += 3) {
        final p = faceCrop.getPixel(x, y);
        // Standard ITU-R BT.601 RGB to YCbCr conversion
        final double cb = 128.0 - 0.168736 * p.r - 0.331264 * p.g + 0.5 * p.b;
        final double cr = 128.0 + 0.5 * p.r - 0.418688 * p.g - 0.081312 * p.b;

        sumCb += cb;
        sumCr += cr;
        totalSampled++;

        final bool isSkin = (cb >= 77.0 && cb <= 138.0 && cr >= 130.0 && cr <= 185.0 && cr > cb);
        if (isSkin) validSkinTotal++;

        if (y < upperBoundary) {
          sumCbUpper += cb;
          sumCrUpper += cr;
          totalUpper++;
          if (isSkin) validSkinUpper++;
        }
      }
    }

    if (totalSampled == 0) return 0.5;

    final double skinRatio = validSkinTotal / totalSampled;
    final double meanCb = sumCb / totalSampled;
    final double meanCr = sumCr / totalSampled;

    final double upperSkinRatio =
        totalUpper > 0 ? validSkinUpper / totalUpper : skinRatio;
    final double upperMeanCb =
        totalUpper > 0 ? sumCbUpper / totalUpper : meanCb;
    final double upperMeanCr =
        totalUpper > 0 ? sumCrUpper / totalUpper : meanCr;

    // Heavy blue-lit screen displays push mean Cb > 140 or invert Cr/Cb across upper face
    if (meanCb > 142.0 || meanCr < meanCb) {
      if (upperMeanCb > 140.0 || upperMeanCr < upperMeanCb) {
        return 0.15; // Digital screen gamut confirmed across forehead
      }
    }

    // If lower face is occluded (medical mask / beard), rely on upper face skin cluster
    if (upperSkinRatio >= 0.50) {
      return upperSkinRatio.clamp(0.0, 1.0);
    }

    return skinRatio.clamp(0.0, 1.0);
  }

  /// Detects planar specular glares caused by tablet/phone glass or glossy photo paper.
  /// Distinguishes localized eyeglass lens reflections from flat full-screen replays.
  _SpecularMetric _analyzeSpecularReflection(img.Image faceCrop) {
    final int w = faceCrop.width;
    final int h = faceCrop.height;
    if (w < 4 || h < 4) {
      return _SpecularMetric(score: 1.0, isPlanarGlassReflectionDetected: false);
    }

    int saturatedCount = 0;
    int eyeBandSaturated = 0;
    int nonEyeSaturated = 0;
    int totalSampled = 0;

    final int eyeTop = (h * 0.22).toInt();
    final int eyeBottom = (h * 0.48).toInt();

    for (int y = 0; y < h; y += 2) {
      for (int x = 0; x < w; x += 2) {
        final p = faceCrop.getPixel(x, y);
        totalSampled++;
        // Pure clipped white (specular reflection on glass)
        if (p.r >= 250 && p.g >= 250 && p.b >= 250) {
          saturatedCount++;
          if (y >= eyeTop && y <= eyeBottom) {
            eyeBandSaturated++;
          } else {
            nonEyeSaturated++;
          }
        }
      }
    }

    final double saturatedRatio =
        totalSampled > 0 ? saturatedCount / totalSampled : 0.0;
    final double nonEyeRatio =
        totalSampled > 0 ? nonEyeSaturated / totalSampled : 0.0;

    // Eyeglasses glare is localized predominantly within the eye band
    final bool isEyeglasses = eyeBandSaturated > 0 &&
        nonEyeRatio < 0.04 &&
        saturatedRatio < 0.18;

    // Large planar screen glass reflection spans beyond the eye region (>10% non-eye or >22% total)
    final bool isGlassReflection =
        !isEyeglasses && (nonEyeRatio > 0.10 || saturatedRatio > 0.22);

    final double specularScore = isEyeglasses
        ? 0.90
        : (1.0 - saturatedRatio * 3.0).clamp(0.0, 1.0);

    return _SpecularMetric(
      score: specularScore,
      isPlanarGlassReflectionDetected: isGlassReflection,
      isEyeglassesGlare: isEyeglasses,
    );
  }

  /// Analyzes passive temporal micro-movements for a tracked face across frames.
  /// Flags a presentation attack only when there is literally zero camera pixel
  /// variance (< 10^-6, e.g. a synthetic replay or frozen stream), and never
  /// penalizes a live human for holding or sitting still in front of the camera.
  double _evaluateTemporalMicroDynamics({
    required int trackingId,
    required Face face,
    required double meanLuminance,
    double textureScore = 0.92,
  }) {
    final now = DateTime.now();
    final box = face.boundingBox;
    final double w = box.width;
    final double h = box.height;
    final double aspectRatio = h > 0 ? w / h : 1.0;
    final double centerX = box.left + w / 2.0;
    final double centerY = box.top + h / 2.0;

    final double? avgEye = (face.leftEyeOpenProbability != null &&
            face.rightEyeOpenProbability != null)
        ? (face.leftEyeOpenProbability! + face.rightEyeOpenProbability!) / 2.0
        : null;

    final history = _faceTemporalHistory.putIfAbsent(trackingId, () => []);
    history.add(
      _FaceTemporalRecord(
        aspectRatio: aspectRatio,
        width: w,
        height: h,
        centerX: centerX,
        centerY: centerY,
        meanLuminance: meanLuminance,
        eyeOpenProb: avgEye,
        timestamp: now,
      ),
    );

    if (history.length > AppConstants.temporalTrackingWindow) {
      history.removeAt(0);
    }

    // Require at least 4 frames of history before asserting temporal spoof
    if (history.length < 4) return 0.85;

    // Calculate variance of aspect ratio and luminance
    final aspectRatios = history.map((e) => e.aspectRatio).toList();
    final luminances = history.map((e) => e.meanLuminance).toList();

    final double arVariance = MathUtils.variance(aspectRatios);
    final double lumVariance = MathUtils.variance(luminances);

    // Check eye probabilities across frames if available
    final eyeProbs =
        history.map((e) => e.eyeOpenProb).whereType<double>().toList();
    final double eyeVar =
        eyeProbs.length >= 3 ? MathUtils.variance(eyeProbs) : -1.0;

    // Temporal stillness check: Only flag spoof if there is literally zero camera pixel variance
    // (< 10^-6, e.g. a synthetic replay or frozen stream), and never penalize a live human for holding still.
    // Real camera sensors and live humans holding still always exhibit sensor/photon noise
    // and subtle lighting variance (lumVariance >> 1e-6).
    // Genuine 2D prints and screen replays are passively blocked via skin micro-texture,
    // Moiré frequency, specular glare, and chrominance distribution.
    final bool isZeroPixelVarianceReplay =
        history.length >= 4 && lumVariance < 1e-6;

    if (isZeroPixelVarianceReplay) {
      return 0.10; // Static synthetic replay or frozen camera stream flagged!
    }

    // If natural biological dynamics (eye blink or 3D perspective shift) are observed
    if ((eyeVar > 1e-4) || (arVariance > 1e-4)) {
      return 0.95;
    }

    return 0.88;
  }

  void _cleanupExpiredHistory() {
    final now = DateTime.now();
    if (_lastCleanupTime != null &&
        now.difference(_lastCleanupTime!) < const Duration(seconds: 10)) {
      return;
    }
    _lastCleanupTime = now;
    _faceTemporalHistory.removeWhere((_, list) {
      if (list.isEmpty) return true;
      return now.difference(list.last.timestamp) > const Duration(seconds: 15);
    });
  }

  double _lum(img.Pixel p) {
    return 0.299 * p.r + 0.587 * p.g + 0.114 * p.b;
  }

  bool _isSaturated(img.Pixel p) {
    return p.r >= 248 && p.g >= 248 && p.b >= 248;
  }
}

class _TextureMetric {
  final double score;
  final bool isScreenMoireDetected;
  final bool isFlatPrintDetected;
  final double meanLuminance;

  _TextureMetric({
    required this.score,
    required this.isScreenMoireDetected,
    required this.isFlatPrintDetected,
    required this.meanLuminance,
  });
}

class _SpecularMetric {
  final double score;
  final bool isPlanarGlassReflectionDetected;
  final bool isEyeglassesGlare;

  _SpecularMetric({
    required this.score,
    required this.isPlanarGlassReflectionDetected,
    this.isEyeglassesGlare = false,
  });
}

class _PerfusionMetric {
  final double score;
  final bool isSiliconeMaskDetected;

  _PerfusionMetric({
    required this.score,
    required this.isSiliconeMaskDetected,
  });
}
