import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:camera/camera.dart';
import 'package:camera_platform_interface/camera_platform_interface.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import 'package:faceattendence/core/constants/app_constants.dart';
import 'package:faceattendence/core/utils/math_utils.dart';
import 'package:faceattendence/data/models/user_model.dart';
import 'package:faceattendence/ml/biometrics_pipeline.dart';
import 'package:faceattendence/ml/face_quality_assessment.dart';
import 'package:faceattendence/ml/liveness_detector.dart';
import 'package:faceattendence/ml/passive_liveness_detector.dart';
import 'package:faceattendence/ml/recognizer_service.dart';

void main() {
  group('Modern MathUtils & Biometrics Utilities', () {
    test('fastCosineDistance matches standard cosine distance on normalized vectors', () {
      final v1 = MathUtils.l2Normalize([0.5, 0.8, -0.3, 0.1]);
      final v2 = MathUtils.l2Normalize([0.4, 0.9, -0.2, 0.05]);

      final distStandard = MathUtils.cosineDistance(v1, v2);
      final distFast = MathUtils.fastCosineDistance(v1, v2);

      expect(distFast, closeTo(distStandard, 1e-5));
    });

    test('fastCosineDistance handles empty or mismatched vectors safely', () {
      expect(MathUtils.fastCosineDistance([], []), equals(2.0));
      expect(MathUtils.fastCosineDistance([1.0], [1.0, 2.0]), equals(2.0));
    });

    test('computeCentroid generates unit vector representing template cluster', () {
      final t1 = MathUtils.l2Normalize([1.0, 0.0, 0.0]);
      final t2 = MathUtils.l2Normalize([0.8, 0.2, 0.0]);
      final t3 = MathUtils.l2Normalize([0.9, -0.1, 0.0]);

      final centroid = MathUtils.computeCentroid([t1, t2, t3]);
      expect(centroid.length, equals(3));

      // Centroid must be unit length
      final norm = MathUtils.dotProduct(centroid, centroid);
      expect(norm, closeTo(1.0, 1e-5));

      // Distance from centroid to cluster members should be small
      expect(MathUtils.fastCosineDistance(centroid, t1), lessThan(0.05));
    });

    test('computeCentroid handles edge cases (empty list, single vector)', () {
      expect(MathUtils.computeCentroid([]), isEmpty);
      final single = [0.0, 3.0, 4.0];
      final cSingle = MathUtils.computeCentroid([single]);
      expect(cSingle[1], closeTo(0.6, 1e-5));
      expect(cSingle[2], closeTo(0.8, 1e-5));
    });

    test('adaptivePhotometricNormalize enhances dynamic range of low contrast image', () {
      final lowContrast = img.Image(width: 16, height: 16);
      // Fill with dull gray values 100 to 120
      for (int y = 0; y < 16; y++) {
        for (int x = 0; x < 16; x++) {
          lowContrast.setPixelRgb(x, y, 100 + x, 100 + x, 100 + x);
        }
      }

      final normalized = MathUtils.adaptivePhotometricNormalize(lowContrast);
      expect(normalized.width, equals(16));
      expect(normalized.height, equals(16));

      // Pixel 0 should be stretched down towards 0, and pixel 15 stretched up towards 255
      final p0 = normalized.getPixel(0, 0);
      final pLast = normalized.getPixel(15, 15);
      expect(p0.r, lessThan(20));
      expect(pLast.r, greaterThan(230));
    });

    test('mean and variance compute accurate statistical moments', () {
      expect(MathUtils.mean([]), equals(0.0));
      expect(MathUtils.variance([]), equals(0.0));

      final values = [10.0, 20.0, 30.0];
      expect(MathUtils.mean(values), closeTo(20.0, 1e-6));
      // Variance for 10, 20, 30 is ((100 + 0 + 100) / 3) = 66.6666...
      expect(MathUtils.variance(values), closeTo(66.6667, 1e-3));
    });
  });

  group('UserModel Multi-Template Serialization', () {
    test('UserModel serializes and deserializes multiple templates accurately', () {
      final user = UserModel(
        id: 42,
        name: 'Alice Johnson',
        embedding: [0.1, 0.2, 0.3],
        templates: [
          [0.1, 0.2, 0.3],
          [0.12, 0.19, 0.31],
        ],
        registeredAt: '2026-10-02T15:30:00Z',
      );

      final map = user.toMap();
      expect(map['templates'], isNotNull);

      final restored = UserModel.fromMap(map);
      expect(restored.id, equals(42));
      expect(restored.name, equals('Alice Johnson'));
      expect(restored.templates, isNotNull);
      expect(restored.templates!.length, equals(2));
      expect(restored.templates![1][0], closeTo(0.12, 1e-5));
    });

    test('UserModel backward compatibility when templates column is absent', () {
      final legacyMap = <String, dynamic>{
        'id': 10,
        'name': 'Bob Smith',
        'embedding': '[0.5, 0.5]',
        'registered_at': '2026-10-01T00:00:00Z',
      };

      final restored = UserModel.fromMap(legacyMap);
      expect(restored.name, equals('Bob Smith'));
      expect(restored.embedding.length, equals(2));
      expect(restored.templates, isNull);
    });
  });

  group('FaceQualityAssessment (FQA) Tests', () {
    final fqa = FaceQualityAssessment();

    img.Image createSyntheticFace({
      int r = 180,
      int g = 140,
      int b = 120,
      bool addTexture = true,
    }) {
      final image = img.Image(width: 112, height: 112);
      for (int y = 0; y < 112; y++) {
        for (int x = 0; x < 112; x++) {
          final noise = addTexture ? ((x * 7 + y * 13) % 25) - 12 : 0;
          image.setPixelRgb(
            x,
            y,
            (r + noise).clamp(0, 255),
            (g + noise).clamp(0, 255),
            (b + noise).clamp(0, 255),
          );
        }
      }
      return image;
    }

    Face createFace({
      double yaw = 0.0,
      double pitch = 0.0,
      double roll = 0.0,
      Rect boundingBox = const Rect.fromLTWH(100, 100, 150, 150),
    }) {
      return Face(
        boundingBox: boundingBox,
        landmarks: {},
        contours: {},
        headEulerAngleX: pitch,
        headEulerAngleY: yaw,
        headEulerAngleZ: roll,
      );
    }

    test('Accepts high quality frontal face with optimal lighting and texture', () {
      final face = createFace(yaw: 5.0, pitch: -3.0, roll: 2.0);
      final faceCrop = createSyntheticFace();

      final result = fqa.assessQuality(
        face: face,
        faceCrop: faceCrop,
        imageSize: const Size(640, 480),
      );

      expect(result.isAcceptable, isTrue);
      expect(result.defect, equals(FaceQualityDefect.none));
      expect(result.overallScore, greaterThan(0.50));
    });

    test('Rejects steep yaw angle (> 35°)', () {
      final face = createFace(yaw: 42.0);
      final faceCrop = createSyntheticFace();

      final result = fqa.assessQuality(
        face: face,
        faceCrop: faceCrop,
        imageSize: const Size(640, 480),
      );

      expect(result.isAcceptable, isFalse);
      expect(result.defect, equals(FaceQualityDefect.steepYaw));
    });

    test('Rejects steep pitch angle (> 25°)', () {
      final face = createFace(pitch: -32.0);
      final faceCrop = createSyntheticFace();

      final result = fqa.assessQuality(
        face: face,
        faceCrop: faceCrop,
        imageSize: const Size(640, 480),
      );

      expect(result.isAcceptable, isFalse);
      expect(result.defect, equals(FaceQualityDefect.steepPitch));
    });

    test('Rejects steep roll angle (> 20°)', () {
      final face = createFace(roll: 26.0);
      final faceCrop = createSyntheticFace();

      final result = fqa.assessQuality(
        face: face,
        faceCrop: faceCrop,
        imageSize: const Size(640, 480),
      );

      expect(result.isAcceptable, isFalse);
      expect(result.defect, equals(FaceQualityDefect.steepRoll));
    });

    test('Rejects severe motion blur (flat Laplacian)', () {
      final face = createFace();
      // Completely flat uniform color image -> zero edge variance / motion blur
      final flatImage = createSyntheticFace(addTexture: false);

      final result = fqa.assessQuality(
        face: face,
        faceCrop: flatImage,
        imageSize: const Size(640, 480),
      );

      expect(result.isAcceptable, isFalse);
      expect(result.defect, equals(FaceQualityDefect.motionBlur));
    });

    test('Rejects severely underexposed (dark) face', () {
      final face = createFace();
      final darkImage = createSyntheticFace(r: 15, g: 15, b: 15);

      final result = fqa.assessQuality(
        face: face,
        faceCrop: darkImage,
        imageSize: const Size(640, 480),
      );

      expect(result.isAcceptable, isFalse);
      expect(result.defect, equals(FaceQualityDefect.underexposed));
    });

    test('Rejects severely overexposed (washed out) face', () {
      final face = createFace();
      final brightImage = createSyntheticFace(r: 252, g: 252, b: 252);

      final result = fqa.assessQuality(
        face: face,
        faceCrop: brightImage,
        imageSize: const Size(640, 480),
      );

      expect(result.isAcceptable, isFalse);
      expect(result.defect, equals(FaceQualityDefect.overexposed));
    });

    test('Rejects face resolution that is too small (< 60px)', () {
      final tinyFace = createFace(
        boundingBox: const Rect.fromLTWH(100, 100, 45, 45),
      );
      final crop = createSyntheticFace();

      final result = fqa.assessQuality(
        face: tinyFace,
        faceCrop: crop,
        imageSize: const Size(640, 480),
      );

      expect(result.isAcceptable, isFalse);
      expect(result.defect, equals(FaceQualityDefect.faceTooSmall));
    });

    test('Rejects face clipped against frame boundaries', () {
      final clippedFace = createFace(
        boundingBox: const Rect.fromLTWH(2, 50, 120, 120),
      );
      final crop = createSyntheticFace();

      final result = fqa.assessQuality(
        face: clippedFace,
        faceCrop: crop,
        imageSize: const Size(640, 480),
      );

      expect(result.isAcceptable, isFalse);
      expect(result.defect, equals(FaceQualityDefect.clippedBoundary));
    });

    test('Accepts face located in vertical lower half of portrait 720x1280 frame without false clippedBoundary rejection', () {
      final portraitFace = createFace(
        boundingBox: const Rect.fromLTWH(180, 550, 220, 250),
      );
      final crop = createSyntheticFace();

      final result = fqa.assessQuality(
        face: portraitFace,
        faceCrop: crop,
        imageSize: const Size(720, 1280),
      );

      expect(result.isAcceptable, isTrue);
      expect(result.defect, equals(FaceQualityDefect.none));
    });
  });

  group('PassiveLivenessDetector (Zero-Friction PAD) Tests', () {
    final pad = PassiveLivenessDetector();

    Face createFace({
      Rect boundingBox = const Rect.fromLTWH(100, 100, 150, 150),
    }) {
      return Face(
        boundingBox: boundingBox,
        landmarks: {},
        contours: {},
      );
    }

    img.Image createSkinFace() {
      // Natural skin with smooth illumination gradient across cheekbones
      final image = img.Image(width: 112, height: 112);
      for (int y = 0; y < 112; y++) {
        for (int x = 0; x < 112; x++) {
          final double grad = (x * 0.8 - y * 0.4);
          final int r = (195 + grad).round().clamp(0, 255);
          final int g = (145 + grad * 0.75).round().clamp(0, 255);
          final int b = (125 + grad * 0.65).round().clamp(0, 255);
          image.setPixelRgb(x, y, r, g, b);
        }
      }
      return image;
    }

    test('Verifies live human face passively with no user interaction required', () {
      pad.reset();
      final face = createFace();
      final skinCrop = createSkinFace();

      final result = pad.evaluateFace(face: face, faceCrop: skinCrop);

      expect(result.isLive, isTrue);
      expect(result.attackType, equals(PassiveAttackType.none));
      expect(result.livenessScore, greaterThanOrEqualTo(AppConstants.passiveLivenessThreshold));
    });

    test('Blocks screen replay with subpixel Moiré grid frequency patterns', () {
      pad.reset();
      final face = createFace();
      final moireCrop = img.Image(width: 112, height: 112);

      // High-frequency alternating pixel grid (Moiré screen artifact)
      for (int y = 0; y < 112; y++) {
        for (int x = 0; x < 112; x++) {
          final isEven = (x + y) % 2 == 0;
          final v = isEven ? 230 : 50;
          moireCrop.setPixelRgb(x, y, v, v, v);
        }
      }

      final result = pad.evaluateFace(face: face, faceCrop: moireCrop);

      expect(result.isLive, isFalse);
      expect(result.attackType, equals(PassiveAttackType.screenReplay));
    });

    test('Blocks planar glass specular reflection (phone / tablet screen glare)', () {
      pad.reset();
      final face = createFace();
      final glareCrop = createSkinFace();

      // Introduce a 25% contiguous clipped white glare patch (typical of phone screen reflection)
      for (int y = 10; y < 70; y++) {
        for (int x = 10; x < 70; x++) {
          glareCrop.setPixelRgb(x, y, 255, 255, 255);
        }
      }

      final result = pad.evaluateFace(face: face, faceCrop: glareCrop);

      expect(result.isLive, isFalse);
      expect(result.attackType, equals(PassiveAttackType.specularGlare));
    });

    test('Blocks electronic screen color distortion with elevated blue gamut', () {
      pad.reset();
      final face = createFace();
      final screenColorCrop = img.Image(width: 112, height: 112);

      // Skewed screen RGB: excess blue (R=100, G=120, B=230 -> Cb > 150)
      for (int y = 0; y < 112; y++) {
        for (int x = 0; x < 112; x++) {
          screenColorCrop.setPixelRgb(x, y, 100, 120, 230);
        }
      }

      final result = pad.evaluateFace(face: face, faceCrop: screenColorCrop);

      expect(result.isLive, isFalse);
    });

    test('Blocks frozen static 2D photo over continuous temporal frames', () {
      pad.reset();
      const trackingId = 999;
      final frozenFace = Face(
        boundingBox: const Rect.fromLTWH(120, 120, 140, 140),
        landmarks: {},
        contours: {},
        trackingId: trackingId,
      );
      final staticCrop = createSkinFace();

      // Feed multiple identical frames (photo held completely stationary)
      PassiveLivenessResult lastResult = pad.evaluateFace(
        face: frozenFace,
        faceCrop: staticCrop,
        trackingId: trackingId,
      );

      for (int i = 0; i < 6; i++) {
        lastResult = pad.evaluateFace(
          face: frozenFace,
          faceCrop: staticCrop,
          trackingId: trackingId,
        );
      }

      expect(lastResult.isLive, isFalse);
      expect(lastResult.attackType, equals(PassiveAttackType.staticPlanarSpoof));
    });

    test('Verifies live human face wearing a surgical mask without false rejection', () {
      pad.reset();
      final face = createFace();
      final maskCrop = img.Image(width: 112, height: 112);

      // Upper half (forehead & eyes): natural living skin
      for (int y = 0; y < 56; y++) {
        for (int x = 0; x < 112; x++) {
          final double grad = (x * 0.8 - y * 0.4);
          final int r = (195 + grad).round().clamp(0, 255);
          final int g = (145 + grad * 0.75).round().clamp(0, 255);
          final int b = (125 + grad * 0.65).round().clamp(0, 255);
          maskCrop.setPixelRgb(x, y, r, g, b);
        }
      }
      // Lower half (nose, mouth, chin): light blue surgical mask with subtle fabric fold gradient
      for (int y = 56; y < 112; y++) {
        for (int x = 0; x < 112; x++) {
          final double grad = (x * 0.4 - y * 0.2);
          maskCrop.setPixelRgb(
            x,
            y,
            (70 + grad).round().clamp(0, 255),
            (160 + grad).round().clamp(0, 255),
            (210 + grad).round().clamp(0, 255),
          );
        }
      }

      final result = pad.evaluateFace(face: face, faceCrop: maskCrop);

      expect(result.isLive, isTrue);
      expect(result.attackType, equals(PassiveAttackType.none));
      expect(result.livenessScore, greaterThanOrEqualTo(AppConstants.passiveLivenessThreshold));
    });

    test('Accepts live face with localized eyeglass lens reflection', () {
      pad.reset();
      final face = createFace();
      final glassesCrop = createSkinFace();

      // Add specular reflections on glasses lenses (localized in eye band y in 30..42)
      for (int y = 30; y < 42; y++) {
        for (int x = 25; x < 40; x++) {
          glassesCrop.setPixelRgb(x, y, 255, 255, 255);
        }
        for (int x = 72; x < 87; x++) {
          glassesCrop.setPixelRgb(x, y, 255, 255, 255);
        }
      }

      final result = pad.evaluateFace(face: face, faceCrop: glassesCrop);

      expect(result.isLive, isTrue);
      expect(result.attackType, equals(PassiveAttackType.none));
      expect(result.livenessScore, greaterThanOrEqualTo(AppConstants.passiveLivenessThreshold));
    });

    test('Blocks synthetic silicone mask with flat perfusion and anomalous pigments', () {
      pad.reset();
      final face = createFace();
      final siliconeCrop = img.Image(width: 112, height: 112);

      // 3D sculpted silicone mask with smooth contour gradient (not flat 2D print),
      // anomalous yellow-tinted pigment (G/R > 0.94), and flat perfusion (deltaCr == 0)
      for (int y = 0; y < 112; y++) {
        for (int x = 0; x < 112; x++) {
          final double grad = (x * 0.6 - y * 0.3);
          siliconeCrop.setPixelRgb(
            x,
            y,
            (210 + grad).round().clamp(0, 255),
            (202 + grad * 0.95).round().clamp(0, 255),
            (140 + grad * 0.65).round().clamp(0, 255),
          );
        }
      }

      final result = pad.evaluateFace(face: face, faceCrop: siliconeCrop);

      expect(result.isLive, isFalse);
      expect(result.attackType, equals(PassiveAttackType.siliconeMask));
    });

    test('Blocks genuine 2D photo printout held with hand tremors over temporal frames via micro-texture', () {
      pad.reset();
      const trackingId = 888;
      // Flat paper printout has suppressed gradient energy (< 0.5 across forehead)
      final photoCrop = img.Image(width: 112, height: 112);
      for (int y = 0; y < 112; y++) {
        for (int x = 0; x < 112; x++) {
          photoCrop.setPixelRgb(x, y, 195, 145, 125);
        }
      }

      // Hand tremor jitter coordinates: centerX/centerY oscillate slightly
      final jitterOffsets = [
        const Offset(0, 0),
        const Offset(2.0, -1.5),
        const Offset(-1.5, 2.0),
        const Offset(1.8, 1.0),
        const Offset(-1.0, -1.8),
        const Offset(1.5, -1.0),
      ];

      PassiveLivenessResult lastResult = pad.evaluateFace(
        face: Face(
          boundingBox: const Rect.fromLTWH(120, 120, 140, 140),
          landmarks: {},
          contours: {},
          leftEyeOpenProbability: 0.99,
          rightEyeOpenProbability: 0.99,
          trackingId: trackingId,
        ),
        faceCrop: photoCrop,
        trackingId: trackingId,
      );

      for (int i = 1; i < jitterOffsets.length; i++) {
        final offset = jitterOffsets[i];
        lastResult = pad.evaluateFace(
          face: Face(
            // Rigid 2D photo translates on screen with hand shaking
            boundingBox: Rect.fromLTWH(
              120 + offset.dx,
              120 + offset.dy,
              140,
              140,
            ),
            landmarks: {},
            contours: {},
            leftEyeOpenProbability: 0.99,
            rightEyeOpenProbability: 0.99,
            trackingId: trackingId,
          ),
          faceCrop: photoCrop,
          trackingId: trackingId,
        );
      }

      expect(lastResult.isLive, isFalse);
      expect(lastResult.attackType, equals(PassiveAttackType.printPhoto2D));
      expect(lastResult.message, contains('2D photo print detected'));
    });

    test('Verifies live human sitting still in front of camera over continuous temporal frames without false spoof lockout', () {
      pad.reset();
      const trackingId = 777;

      for (int i = 0; i < 10; i++) {
        // Natural camera sensor noise causes subtle frame-to-frame luminance fluctuation (variance >> 1e-6)
        final double noise = (i % 2 == 0 ? 1.0 : -1.0) * (1.2 + (i * 0.1));
        final liveCrop = img.Image(width: 112, height: 112);
        for (int y = 0; y < 112; y++) {
          for (int x = 0; x < 112; x++) {
            final double grad = (x * 0.8 - y * 0.4);
            final int r = (195 + grad + noise).round().clamp(0, 255);
            final int g = (145 + grad * 0.75 + noise).round().clamp(0, 255);
            final int b = (125 + grad * 0.65 + noise).round().clamp(0, 255);
            liveCrop.setPixelRgb(x, y, r, g, b);
          }
        }

        final result = pad.evaluateFace(
          face: Face(
            boundingBox: const Rect.fromLTWH(120, 120, 140, 140),
            landmarks: {},
            contours: {},
            leftEyeOpenProbability: 0.98,
            rightEyeOpenProbability: 0.98,
            trackingId: trackingId,
          ),
          faceCrop: liveCrop,
          trackingId: trackingId,
        );

        expect(result.isLive, isTrue,
            reason: 'Frame $i must not falsely lock out live user sitting still');
        expect(result.attackType, equals(PassiveAttackType.none));
        expect(result.message, equals('Live Face Verified ✓'));
      }
    });

    test('Verifies live human sitting still with ML Kit bounding box jitter and steady gaze without false lockout', () {
      pad.reset();
      const trackingId = 778;

      // Realistic bounding box jitter from ML Kit running on consecutive camera frames
      final jitterOffsets = [
        const Offset(0, 0),
        const Offset(1.2, -0.8),
        const Offset(-0.8, 1.0),
        const Offset(1.0, 0.5),
        const Offset(-0.5, -1.0),
        const Offset(0.8, -0.6),
        const Offset(-1.0, 0.8),
        const Offset(0.5, 0.5),
      ];

      for (int i = 0; i < jitterOffsets.length; i++) {
        final offset = jitterOffsets[i];
        final double noise = (i % 2 == 0 ? 1.5 : -1.5);
        final liveCrop = img.Image(width: 112, height: 112);
        for (int y = 0; y < 112; y++) {
          for (int x = 0; x < 112; x++) {
            final double grad = (x * 0.8 - y * 0.4);
            final int r = (195 + grad + noise).round().clamp(0, 255);
            final int g = (145 + grad * 0.75 + noise).round().clamp(0, 255);
            final int b = (125 + grad * 0.65 + noise).round().clamp(0, 255);
            liveCrop.setPixelRgb(x, y, r, g, b);
          }
        }

        final result = pad.evaluateFace(
          face: Face(
            boundingBox: Rect.fromLTWH(
              120 + offset.dx,
              120 + offset.dy,
              140,
              140,
            ),
            landmarks: {},
            contours: {},
            leftEyeOpenProbability: 0.98,
            rightEyeOpenProbability: 0.98,
            trackingId: trackingId,
          ),
          faceCrop: liveCrop,
          trackingId: trackingId,
        );

        expect(result.isLive, isTrue,
            reason: 'Frame $i with ML Kit box jitter must not falsely flag spoof');
        expect(result.attackType, equals(PassiveAttackType.none));
      }
    });

    test('Blocks genuine 2D photo printout via micro-texture gradient suppression', () {
      pad.reset();
      final face = createFace();

      // Flat paper printout has suppressed gradient energy (< 0.5 across forehead)
      final printCrop = img.Image(width: 112, height: 112);
      for (int y = 0; y < 112; y++) {
        for (int x = 0; x < 112; x++) {
          // Flat uniform skin color with zero micro-pores/texture
          printCrop.setPixelRgb(x, y, 195, 145, 125);
        }
      }

      final result = pad.evaluateFace(face: face, faceCrop: printCrop);

      expect(result.isLive, isFalse);
      expect(result.attackType, equals(PassiveAttackType.printPhoto2D));
      expect(result.message, contains('2D photo print detected'));
    });

    test('Blocks zero-pixel-variance synthetic replay even when ML Kit bounding box has aspect ratio jitter', () {
      pad.reset();
      const trackingId = 998;
      final replayCrop = createSkinFace();

      PassiveLivenessResult lastResult = pad.evaluateFace(
        face: Face(
          boundingBox: const Rect.fromLTWH(120, 120, 140, 140),
          landmarks: {},
          contours: {},
          trackingId: trackingId,
        ),
        faceCrop: replayCrop,
        trackingId: trackingId,
      );

      // Camera pixels are 100% frozen/identical, but bounding box aspect ratio fluctuates slightly
      final aspectJitters = [
        const Rect.fromLTWH(120, 120, 140, 140),
        const Rect.fromLTWH(120, 120, 140.5, 139.5),
        const Rect.fromLTWH(120, 120, 139.5, 140.5),
        const Rect.fromLTWH(120, 120, 140.2, 139.8),
        const Rect.fromLTWH(120, 120, 139.8, 140.2),
      ];

      for (final box in aspectJitters) {
        lastResult = pad.evaluateFace(
          face: Face(
            boundingBox: box,
            landmarks: {},
            contours: {},
            trackingId: trackingId,
          ),
          faceCrop: replayCrop,
          trackingId: trackingId,
        );
      }

      expect(lastResult.isLive, isFalse);
      expect(lastResult.attackType, equals(PassiveAttackType.staticPlanarSpoof));
      expect(lastResult.message, contains('Static photo spoof detected'));
    });

    test('Blocks 2D photo printout with sensor noise and hand tremor continuously across 10 frames via micro-texture', () {
      pad.reset();
      const trackingId = 887;

      for (int i = 0; i < 10; i++) {
        final double noise = (i % 2 == 0 ? 0.3 : -0.3);
        final printCrop = img.Image(width: 112, height: 112);
        for (int y = 0; y < 112; y++) {
          for (int x = 0; x < 112; x++) {
            final int r = (195 + noise).round().clamp(0, 255);
            final int g = (145 + noise).round().clamp(0, 255);
            final int b = (125 + noise).round().clamp(0, 255);
            printCrop.setPixelRgb(x, y, r, g, b);
          }
        }

        final result = pad.evaluateFace(
          face: Face(
            boundingBox: Rect.fromLTWH(120 + (i % 3).toDouble(), 120 + (i % 2).toDouble(), 140, 140),
            landmarks: {},
            contours: {},
            leftEyeOpenProbability: 0.99,
            rightEyeOpenProbability: 0.99,
            trackingId: trackingId,
          ),
          faceCrop: printCrop,
          trackingId: trackingId,
        );

        expect(result.isLive, isFalse, reason: 'Frame $i of 2D print must be blocked');
        expect(result.attackType, equals(PassiveAttackType.printPhoto2D));
      }
    });

    test('Verifies live user sitting completely motionless for 30 consecutive frames without false spoof lockout', () {
      pad.reset();
      const trackingId = 776;

      for (int i = 0; i < 30; i++) {
        // Subtle real CMOS sensor noise (lumVariance ~ 0.5 - 2.0 >> 1e-6)
        final double noise = (i % 2 == 0 ? 0.8 : -0.8) + (i % 3) * 0.2;
        final liveCrop = img.Image(width: 112, height: 112);
        for (int y = 0; y < 112; y++) {
          for (int x = 0; x < 112; x++) {
            final double grad = (x * 0.8 - y * 0.4);
            final int r = (195 + grad + noise).round().clamp(0, 255);
            final int g = (145 + grad * 0.75 + noise).round().clamp(0, 255);
            final int b = (125 + grad * 0.65 + noise).round().clamp(0, 255);
            liveCrop.setPixelRgb(x, y, r, g, b);
          }
        }

        final result = pad.evaluateFace(
          face: Face(
            boundingBox: const Rect.fromLTWH(120, 120, 140, 140),
            landmarks: {},
            contours: {},
            leftEyeOpenProbability: 0.95,
            rightEyeOpenProbability: 0.95,
            trackingId: trackingId,
          ),
          faceCrop: liveCrop,
          trackingId: trackingId,
        );

        expect(result.isLive, isTrue,
            reason: 'Frame $i of live user motionless must not be locked out');
        expect(result.attackType, equals(PassiveAttackType.none));
        expect(result.message, equals('Live Face Verified ✓'));
      }
    });
  });

  group('RecognizerService Multi-Template Matching Tests', () {
    final recognizer = RecognizerService();

    test('Matches user against multi-view templates when frontal differs from profile', () {
      // User enrolled with two views: frontal (vFrontal) and 15-degree yaw (vYaw)
      final vFrontal = MathUtils.l2Normalize(List.generate(512, (i) => i.toDouble()));
      final vYaw = MathUtils.l2Normalize(List.generate(512, (i) => (i * 2 + 5).toDouble()));
      final centroid = MathUtils.computeCentroid([vFrontal, vYaw]);

      final user = UserModel(
        id: 1,
        name: 'Rahul Sharma',
        embedding: centroid,
        templates: [vFrontal, vYaw],
      );

      // Query vector that is close to the vYaw template
      final queryYaw = MathUtils.l2Normalize(
        List.generate(512, (i) => (i * 2 + 5.2).toDouble()),
      );

      final match = recognizer.findBestMatch(queryYaw, [user]);

      expect(match.isMatch, isTrue);
      expect(match.user?.name, equals('Rahul Sharma'));
      expect(match.distance, lessThan(AppConstants.matchThreshold));
    });

    test('Rejects unrecognized intruder', () {
      final genuine = MathUtils.l2Normalize(List.generate(512, (i) => 1.0));
      final user = UserModel(id: 1, name: 'Employee', embedding: genuine);

      // Intruder orthogonal/opposite vector
      final intruder = MathUtils.l2Normalize(
        List.generate(512, (i) => i % 2 == 0 ? 1.0 : -1.0),
      );

      final match = recognizer.findBestMatch(intruder, [user]);
      expect(match.isMatch, isFalse);
      expect(match.user, isNull);
    });

    test('Gracefully handles empty enrolled list', () {
      final query = MathUtils.l2Normalize([1.0, 2.0, 3.0]);
      final match = recognizer.findBestMatch(query, []);
      expect(match.isMatch, isFalse);
      expect(match.distance, equals(2.0));
      expect(match.user, isNull);
    });

    test('Dual-pass photometric fallback recovers match under harsh lighting/shadows', () {
      // Create orthonormal basis vectors u and w
      final u = MathUtils.l2Normalize(List.generate(512, (i) => i.toDouble()));
      final rawW = List.generate(512, (i) => (511 - i).toDouble());
      final dotUW = MathUtils.dotProduct(u, rawW);
      final w = MathUtils.l2Normalize(List.generate(512, (i) => rawW[i] - dotUW * u[i]));

      final user = UserModel(
        id: 101,
        name: 'Carlos Gomez',
        embedding: u,
      );

      // Primary query vector with cos(theta) = 0.55 -> cosine distance = 0.45 (> matchThreshold 0.40)
      const double cosTheta = 0.55;
      final double sinTheta = sqrt(1.0 - cosTheta * cosTheta);
      final degradedEmbedding = List.generate(
        512,
        (i) => cosTheta * u[i] + sinTheta * w[i],
      );

      // Standard single-pass search fails (0.45 > 0.40)
      final singlePassMatch = recognizer.findBestMatch(degradedEmbedding, [user]);
      expect(singlePassMatch.isMatch, isFalse);
      expect(singlePassMatch.distance, greaterThan(AppConstants.matchThreshold));
      expect(singlePassMatch.distance, lessThanOrEqualTo(AppConstants.matchThreshold + 0.08));

      // Dual-pass with photometrically normalized alternate embedding (distance = 0.15 < 0.40) restores true match
      const double altCosTheta = 0.85;
      final double altSinTheta = sqrt(1.0 - altCosTheta * altCosTheta);
      final normalizedAltEmbedding = List.generate(
        512,
        (i) => altCosTheta * u[i] + altSinTheta * w[i],
      );

      final dualPassMatch = recognizer.findBestMatch(
        degradedEmbedding,
        [user],
        alternateEmbedding: normalizedAltEmbedding,
      );

      expect(dualPassMatch.isMatch, isTrue);
      expect(dualPassMatch.user?.id, equals(101));
      expect(dualPassMatch.distance, lessThan(AppConstants.matchThreshold));
    });
  });

  group('BiometricsPipeline Orchestration Tests', () {
    test('Pipeline initializes subcomponents and manages state correctly', () {
      final pipeline = BiometricsPipeline();
      expect(pipeline.qualityAssessment, isNotNull);
      expect(pipeline.passiveLivenessDetector, isNotNull);
      expect(pipeline.recognizerService, isNotNull);

      // Verify reset and clearCooldowns execute without errors
      pipeline.reset();
      pipeline.clearCooldowns();
    });

    test('Queue prioritization selects foreground face when another person queues behind', () {
      final foregroundFace = Face(
        boundingBox: const Rect.fromLTWH(100, 100, 220, 220), // Area = 48,400
        landmarks: {},
        contours: {},
      );

      final queueBehindFace = Face(
        boundingBox: const Rect.fromLTWH(350, 80, 90, 90), // Area = 8,100
        landmarks: {},
        contours: {},
      );

      // When foreground face is prominent, queueBehind is not flagged as multipleFaces conflict
      final faces = [queueBehindFace, foregroundFace]; // Out of order

      // Gate 1 queue sorting logic
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

      final isCompeting = secondaryArea > 0.55 * primaryArea && secondaryArea >= 70 * 70;
      expect(isCompeting, isFalse);
      expect(sortedFaces.first, equals(foregroundFace));
    });

    test('Queue prioritization detects side-by-side multiple faces (tailgating attempt)', () {
      final person1 = Face(
        boundingBox: const Rect.fromLTWH(50, 100, 180, 180), // Area = 32,400
        landmarks: {},
        contours: {},
      );

      final person2 = Face(
        boundingBox: const Rect.fromLTWH(260, 100, 170, 170), // Area = 28,900
        landmarks: {},
        contours: {},
      );

      final faces = [person1, person2];

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

      final isCompeting = secondaryArea > 0.55 * primaryArea && secondaryArea >= 70 * 70;
      expect(isCompeting, isTrue);
    });
  });

  group('Blink-To-Recognize Liveness & Anti-Spoofing Tests', () {
    Face createMockFace({
      double leftOpen = 0.9,
      double rightOpen = 0.9,
      int? trackingId,
    }) {
      return Face(
        boundingBox: const Rect.fromLTWH(100, 100, 150, 150),
        landmarks: {},
        contours: {},
        leftEyeOpenProbability: leftOpen,
        rightEyeOpenProbability: rightOpen,
        trackingId: trackingId,
      );
    }

    test('Blink cycle transitions smoothly through open -> closed -> open with proper guidance', () {
      final detector = LivenessDetector();

      // 1. User enters frame with eyes open
      final r1 = detector.processFace(createMockFace(leftOpen: 0.92, rightOpen: 0.88));
      expect(r1.isLivenessConfirmed, isFalse);
      expect(r1.phase, equals(BlinkPhase.eyesOpen));
      expect(r1.message, equals('Blink your eyes to check in'));
      expect(r1.progress, equals(0.35));

      // 2. User blinks (eyes close)
      final r2 = detector.processFace(createMockFace(leftOpen: 0.12, rightOpen: 0.10));
      expect(r2.isLivenessConfirmed, isFalse);
      expect(r2.phase, equals(BlinkPhase.eyesClosed));
      expect(r2.message, equals('Blink registered! Open eyes...'));
      expect(r2.progress, equals(0.70));

      // 3. User reopens eyes (blink completed)
      final r3 = detector.processFace(createMockFace(leftOpen: 0.90, rightOpen: 0.95));
      expect(r3.isLivenessConfirmed, isTrue);
      expect(r3.phase, equals(BlinkPhase.blinkCompleted));
      expect(r3.message, equals('Liveness Verified! ✓'));
      expect(r3.progress, equals(1.0));
    });

    test('Single-eye wink does NOT advance to eyesClosed phase', () {
      final detector = LivenessDetector();

      // Open eyes
      detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
      expect(detector.currentPhase, equals(BlinkPhase.eyesOpen));

      // One eye closes (wink), other remains open
      final winkResult = detector.processFace(createMockFace(leftOpen: 0.05, rightOpen: 0.95));
      expect(winkResult.isLivenessConfirmed, isFalse);
      expect(detector.currentPhase, equals(BlinkPhase.eyesOpen), reason: 'Bilateral blink required');
      expect(winkResult.message, equals('Blink your eyes to check in'));
    });

    test('Static photo spoof never confirms liveness and never triggers false lockout', () {
      final detector = LivenessDetector();

      // A static printed photo or display screen held in front of the camera for 30 consecutive frames
      for (int i = 0; i < 30; i++) {
        final result = detector.processFace(createMockFace(leftOpen: 0.95, rightOpen: 0.95));
        expect(result.isLivenessConfirmed, isFalse, reason: 'Static photo must never confirm liveness on frame $i');
        expect(result.phase, equals(BlinkPhase.eyesOpen));
        expect(result.isStaticSpoofSuspected, isFalse, reason: 'Must not trigger false spoof lockout alarm during live framing');
        expect(result.message, equals('Blink your eyes to check in'));
      }
    });

    test('Tracking ID change resets blink phase for the next person', () {
      final detector = LivenessDetector();

      // Person 1 (trackingId: 101) completes a blink
      detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 101));
      detector.processFace(createMockFace(leftOpen: 0.1, rightOpen: 0.1, trackingId: 101));
      final p1Done = detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 101));
      expect(p1Done.isLivenessConfirmed, isTrue);
      expect(detector.currentPhase, equals(BlinkPhase.blinkCompleted));

      // Person 2 (trackingId: 102) enters frame with eyes open
      final p2Start = detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 102));
      expect(p2Start.isLivenessConfirmed, isFalse, reason: 'New trackingId must reset and require a fresh blink');
      expect(p2Start.phase, equals(BlinkPhase.eyesOpen));
      expect(p2Start.message, equals('Blink your eyes to check in'));
    });

    test('Walk-through cooldown is set to 30 seconds', () {
      expect(AppConstants.walkThroughCooldown, equals(const Duration(seconds: 30)));
    });

    test('Pipeline integrates LivenessDetector and resets cleanly', () {
      final pipeline = BiometricsPipeline();
      expect(pipeline.livenessDetector, isNotNull);
      expect(pipeline.livenessDetector.currentPhase, equals(BlinkPhase.waitingForOpen));

      // Simulate a blink on pipeline's detector
      pipeline.livenessDetector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
      expect(pipeline.livenessDetector.currentPhase, equals(BlinkPhase.eyesOpen));

      // Pipeline reset must reset blink phase
      pipeline.reset();
      expect(pipeline.livenessDetector.currentPhase, equals(BlinkPhase.waitingForOpen));
    });

    test('Eyes closed longer than 2000ms resets to waitingForOpen', () async {
      final detector = LivenessDetector();

      // Open eyes
      detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
      expect(detector.currentPhase, equals(BlinkPhase.eyesOpen));

      // Close eyes
      final closedRes = detector.processFace(createMockFace(leftOpen: 0.1, rightOpen: 0.1));
      expect(detector.currentPhase, equals(BlinkPhase.eyesClosed));
      expect(closedRes.message, equals('Blink registered! Open eyes...'));

      // Wait past the 2000ms closed timeout
      await Future.delayed(const Duration(milliseconds: 2100));

      final timeoutRes = detector.processFace(createMockFace(leftOpen: 0.1, rightOpen: 0.1));
      expect(detector.currentPhase, equals(BlinkPhase.waitingForOpen));
      expect(timeoutRes.message, equals('Blink your eyes to check in'));
    });

    test('Null ML Kit eye open probabilities return waitingForOpen with prompt', () {
      final detector = LivenessDetector();
      final nullProbFace = Face(
        boundingBox: const Rect.fromLTWH(100, 100, 150, 150),
        landmarks: {},
        contours: {},
        leftEyeOpenProbability: null,
        rightEyeOpenProbability: null,
      );

      final result = detector.processFace(nullProbFace);
      expect(result.isLivenessConfirmed, isFalse);
      expect(result.phase, equals(BlinkPhase.waitingForOpen));
      expect(result.message, equals('Blink your eyes to check in'));
    });

    test('Bilateral open requirement rejects initial asymmetric wink', () {
      final detector = LivenessDetector();

      // Enters with one eye open (0.95) and one eye closed/squinting (0.25)
      final winkFace = createMockFace(leftOpen: 0.95, rightOpen: 0.25);
      final result = detector.processFace(winkFace);
      expect(result.isLivenessConfirmed, isFalse);
      expect(detector.currentPhase, equals(BlinkPhase.waitingForOpen),
          reason: 'Initial asymmetric wink must not transition to eyesOpen');
      expect(result.message, equals('Blink your eyes to check in'));
    });

    test('Bilateral reopening requirement rejects asymmetric wink on reopening', () {
      final detector = LivenessDetector();

      // Open eyes
      detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
      expect(detector.currentPhase, equals(BlinkPhase.eyesOpen));

      // Close both eyes
      detector.processFace(createMockFace(leftOpen: 0.1, rightOpen: 0.1));
      expect(detector.currentPhase, equals(BlinkPhase.eyesClosed));

      // Reopen only ONE eye (winking)
      final winkReopen = detector.processFace(createMockFace(leftOpen: 0.95, rightOpen: 0.25));
      expect(winkReopen.isLivenessConfirmed, isFalse);
      expect(detector.currentPhase, equals(BlinkPhase.eyesClosed),
          reason: 'Asymmetric reopening wink must NOT advance to blinkCompleted');
      expect(winkReopen.message, equals('Blink registered! Open eyes...'));

      // Now reopen BOTH eyes fully
      final fullReopen = detector.processFace(createMockFace(leftOpen: 0.95, rightOpen: 0.90));
      expect(fullReopen.isLivenessConfirmed, isTrue);
      expect(detector.currentPhase, equals(BlinkPhase.blinkCompleted));
      expect(fullReopen.message, equals('Liveness Verified! ✓'));
    });

    test('Completed phase timeout resets unconsumed blink after 1500ms', () async {
      final detector = LivenessDetector();

      detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
      detector.processFace(createMockFace(leftOpen: 0.1, rightOpen: 0.1));
      final completed = detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
      expect(completed.isLivenessConfirmed, isTrue);
      expect(detector.currentPhase, equals(BlinkPhase.blinkCompleted));

      // Wait past the 1500ms completed timeout
      await Future.delayed(const Duration(milliseconds: 1600));

      final timeoutResult = detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
      expect(timeoutResult.isLivenessConfirmed, isFalse);
      expect(detector.currentPhase, equals(BlinkPhase.waitingForOpen));
      expect(timeoutResult.message, equals('Blink your eyes to check in'));
    });
  });

  group('BiometricsPipeline End-to-End Cooldown & Blink Orchestration Tests', () {
    CameraImage createMockCameraImage({int w = 160, int h = 160}) {
      final bytes = Uint8List(w * h * 3 ~/ 2);
      for (int i = 0; i < bytes.length; i++) {
        bytes[i] = ((i * 17) % 200) + 28;
      }

      return CameraImage.fromPlatformInterface(
        CameraImageData(
          format: const CameraImageFormat(ImageFormatGroup.nv21, raw: 17),
          planes: <CameraImagePlane>[
            CameraImagePlane(
              bytes: bytes,
              bytesPerRow: w,
              width: w,
              height: h,
            ),
          ],
          width: w,
          height: h,
        ),
      );
    }

    const mockCamera = CameraDescription(
      name: 'front',
      lensDirection: CameraLensDirection.front,
      sensorOrientation: 0,
    );

    Face createPipelineMockFace({
      double leftOpen = 0.9,
      double rightOpen = 0.9,
      int? trackingId,
    }) {
      return Face(
        boundingBox: const Rect.fromLTWH(20, 20, 100, 100),
        landmarks: {},
        contours: {},
        leftEyeOpenProbability: leftOpen,
        rightEyeOpenProbability: rightOpen,
        headEulerAngleX: 0.0,
        headEulerAngleY: 0.0,
        headEulerAngleZ: 0.0,
        trackingId: trackingId,
      );
    }

    test('Full Pipeline: Blink triggers single recognition, marks attendance, enforces 30s tracked cooldown without re-inferring', () async {
      final fakeRecognizer = _TestRecognizerService();
      final aliceEmbedding = MathUtils.l2Normalize(List.generate(512, (i) => i.toDouble()));
      fakeRecognizer.embeddingToReturn = aliceEmbedding;

      final enrolledAlice = UserModel(
        id: 1,
        name: 'Alice Smith',
        embedding: aliceEmbedding,
      );

      final pipeline = BiometricsPipeline(recognizerService: fakeRecognizer);
      final cameraImage = createMockCameraImage();

      // 1. Frame 1: Alice enters frame with eyes open (trackingId: 101)
      final f1 = await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 101)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      expect(f1.status, equals(BiometricsStatus.waitingForBlink));
      expect(f1.statusMessage, equals('Blink your eyes to check in'));
      expect(f1.isAttendanceMarked, isFalse);
      expect(fakeRecognizer.predictCallCount, equals(0));

      // 2. Frame 2: Alice blinks (eyes closed)
      final f2 = await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.1, rightOpen: 0.1, trackingId: 101)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      expect(f2.status, equals(BiometricsStatus.waitingForBlink));
      expect(f2.statusMessage, equals('Blink registered! Open eyes...'));
      expect(fakeRecognizer.predictCallCount, equals(0));

      // 3. Frame 3: Alice reopens eyes (sharp eyes-open frame -> recognition executes!)
      final f3 = await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 101)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      expect(f3.status, equals(BiometricsStatus.recognized));
      expect(f3.statusMessage, equals('Welcome, Alice Smith!'));
      expect(f3.isAttendanceMarked, isTrue);
      expect(fakeRecognizer.predictCallCount, equals(1));
      // Blink detector must be cleanly reset after consuming blink
      expect(pipeline.livenessDetector.currentPhase, equals(BlinkPhase.waitingForOpen));

      // 4. Frame 4: Alice remains in front of the camera (same trackingId: 101)
      final f4 = await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 101)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      expect(f4.status, equals(BiometricsStatus.cooldownActive));
      expect(f4.statusMessage, contains('Already checked in: Alice Smith'));
      expect(f4.statusMessage, contains('cooldown'));
      expect(f4.isAttendanceMarked, isFalse);
      // Crucial: Zero TFLite inference run on cooldown frame
      expect(fakeRecognizer.predictCallCount, equals(1));

      // 5. Frame 5: Alice walks away (faces is empty)
      final f5 = await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      expect(f5.status, equals(BiometricsStatus.idle));
      expect(f5.statusMessage, equals('Blink your eyes to check in'));

      // 6. Frame 6: Different person enters (trackingId: 102) -> NOT in cooldown!
      final f6 = await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 102)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      expect(f6.status, equals(BiometricsStatus.waitingForBlink));
      expect(f6.statusMessage, equals('Blink your eyes to check in'));

      // 7. Frame 7: Alice returns 5 seconds later with new trackingId (103)
      // Completes a blink cycle
      await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 103)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.1, rightOpen: 0.1, trackingId: 103)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      final f7 = await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 103)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      // Pipeline recognizes Alice but rejects duplicate attendance with cooldownActive!
      expect(f7.status, equals(BiometricsStatus.cooldownActive));
      expect(f7.statusMessage, contains('Already checked in: Alice Smith'));
      expect(f7.isAttendanceMarked, isFalse);

      // 8. clearCooldowns clears the 30s block and allows Alice to check in again
      pipeline.clearCooldowns();
      await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 104)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.1, rightOpen: 0.1, trackingId: 104)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      final f8 = await pipeline.processFrame(
        cameraImage: cameraImage,
        faces: [createPipelineMockFace(leftOpen: 0.9, rightOpen: 0.9, trackingId: 104)],
        camera: mockCamera,
        enrolledUsers: [enrolledAlice],
      );
      expect(f8.status, equals(BiometricsStatus.recognized));
      expect(f8.isAttendanceMarked, isTrue);
    });
  });
}

class _TestRecognizerService extends RecognizerService {
  List<double> embeddingToReturn = List.filled(512, 0.0);
  int predictCallCount = 0;

  @override
  int get inputWidth => 112;

  @override
  int get inputHeight => 112;

  @override
  List<double> predict(img.Image croppedFace) {
    predictCallCount++;
    return embeddingToReturn;
  }
}
