import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:faceattendence/core/constants/app_constants.dart';
import 'package:faceattendence/core/utils/math_utils.dart';
import 'package:faceattendence/data/models/user_model.dart';
import 'package:faceattendence/ml/liveness_detector.dart';

void main() {
  group('MathUtils Tests', () {
    test('Cosine distance of identical vectors is 0.0', () {
      final v1 = [1.0, 2.0, 3.0, 4.0];
      final v2 = [1.0, 2.0, 3.0, 4.0];

      final distance = MathUtils.cosineDistance(v1, v2);
      expect(distance, closeTo(0.0, 1e-6));
    });

    test('Cosine distance of orthogonal vectors is 1.0', () {
      final v1 = [1.0, 0.0];
      final v2 = [0.0, 1.0];

      final distance = MathUtils.cosineDistance(v1, v2);
      expect(distance, closeTo(1.0, 1e-6));
    });

    test('Cosine distance of opposite vectors is 2.0', () {
      final v1 = [1.0, 2.0];
      final v2 = [-1.0, -2.0];

      final distance = MathUtils.cosineDistance(v1, v2);
      expect(distance, closeTo(2.0, 1e-6));
    });

    test('Cosine distance of different length vectors returns 2.0', () {
      final v1 = [1.0, 2.0];
      final v2 = [1.0, 2.0, 3.0];

      final distance = MathUtils.cosineDistance(v1, v2);
      expect(distance, equals(2.0));
    });

    test('L2 normalization produces unit norm vector', () {
      final vector = [3.0, 4.0]; // Norm is 5.0
      final normalized = MathUtils.l2Normalize(vector);

      expect(normalized[0], closeTo(0.6, 1e-6));
      expect(normalized[1], closeTo(0.8, 1e-6));

      final norm = sqrt(normalized[0] * normalized[0] + normalized[1] * normalized[1]);
      expect(norm, closeTo(1.0, 1e-6));
    });

    test('Match threshold evaluates correctly for 512-D vectors', () {
      // Create a 512-D normalized vector
      final base = List<double>.generate(512, (i) => i.toDouble());
      final normBase = MathUtils.l2Normalize(base);

      // Create a slightly perturbed vector
      final perturbed = List<double>.generate(512, (i) => i.toDouble() + 0.1);
      final normPerturbed = MathUtils.l2Normalize(perturbed);

      final distance = MathUtils.cosineDistance(normBase, normPerturbed);
      expect(distance, lessThan(AppConstants.matchThreshold));
    });
  });

  group('UserModel Serialization Tests', () {
    test('UserModel converts to and from Map accurately', () {
      final original = UserModel(
        id: 1,
        name: 'John Doe',
        embedding: [0.123, 0.456, -0.789],
        registeredAt: '2026-10-02T10:00:00Z',
      );

      final map = original.toMap();
      final restored = UserModel.fromMap(map);

      expect(restored.id, equals(original.id));
      expect(restored.name, equals(original.name));
      expect(restored.embedding.length, equals(original.embedding.length));
      expect(restored.embedding[0], closeTo(0.123, 1e-6));
      expect(restored.embedding[1], closeTo(0.456, 1e-6));
      expect(restored.embedding[2], closeTo(-0.789, 1e-6));
      expect(restored.registeredAt, equals(original.registeredAt));
    });
  });

  group('LivenessDetector (Anti-Spoofing) Tests', () {
    Face createMockFace({double leftOpen = 0.9, double rightOpen = 0.9}) {
      return Face(
        boundingBox: Rect.zero,
        landmarks: {},
        contours: {},
        leftEyeOpenProbability: leftOpen,
        rightEyeOpenProbability: rightOpen,
      );
    }

    test('A complete Open -> Closed -> Open blink cycle confirms liveness', () {
      final detector = LivenessDetector();

      // Step 1: Open eyes (user aligns face)
      final r1 = detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
      expect(r1.isLivenessConfirmed, isFalse);
      expect(detector.currentPhase, equals(BlinkPhase.eyesOpen));

      // Step 2: Eyes closed during natural blink
      final r2 = detector.processFace(createMockFace(leftOpen: 0.1, rightOpen: 0.1));
      expect(r2.isLivenessConfirmed, isFalse);
      expect(detector.currentPhase, equals(BlinkPhase.eyesClosed));

      // Step 3: Eyes reopened -> Liveness verified!
      final r3 = detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
      expect(r3.isLivenessConfirmed, isTrue);
      expect(detector.currentPhase, equals(BlinkPhase.blinkCompleted));
    });

    test('Static eyes open (e.g. printed paper photo) does NOT confirm liveness', () {
      final detector = LivenessDetector();

      // Repeated frames with permanently open eyes (static photo)
      for (int i = 0; i < 5; i++) {
        final r = detector.processFace(createMockFace(leftOpen: 0.9, rightOpen: 0.9));
        expect(r.isLivenessConfirmed, isFalse);
      }
      expect(detector.currentPhase, equals(BlinkPhase.eyesOpen));
    });
  });
}
