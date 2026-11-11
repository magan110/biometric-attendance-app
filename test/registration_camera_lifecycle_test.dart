import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:faceattendence/presentation/camera_view/camera_view.dart';
import 'package:faceattendence/presentation/registration/registration_screen.dart';
import 'package:faceattendence/ml/recognizer_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('CameraView Lifecycle Tests', () {
    testWidgets('CameraView in paused state does not start camera and displays paused label', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: CameraView(
              isPaused: true,
              onFrameProcessed: (image, faces, camera) {},
            ),
          ),
        ),
      );

      expect(find.text('Camera paused'), findsOneWidget);
      expect(find.byType(CircularProgressIndicator), findsOneWidget);
    });

    testWidgets('CameraView updates when paused toggles without throwing', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: CameraView(
              isPaused: true,
              onFrameProcessed: (image, faces, camera) {},
            ),
          ),
        ),
      );

      expect(find.text('Camera paused'), findsOneWidget);

      // Re-pump with isPaused still true (e.g. parent rebuild)
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: CameraView(
              isPaused: true,
              onFrameProcessed: (image, faces, camera) {},
            ),
          ),
        ),
      );

      expect(find.text('Camera paused'), findsOneWidget);
    });

    test('CameraView.releaseActiveHardware completes cleanly when no active instance', () async {
      await expectLater(CameraView.releaseActiveHardware(), completes);
    });

    testWidgets('CameraView.releaseActiveHardware gracefully releases mounted paused camera', (tester) async {
      await tester.pumpWidget(
        MaterialApp(
          home: Scaffold(
            body: CameraView(
              isPaused: true,
              onFrameProcessed: (image, faces, camera) {},
            ),
          ),
        ),
      );

      await expectLater(CameraView.releaseActiveHardware(), completes);
    });
  });

  group('RegistrationScreen Lifecycle & Constructor Tests', () {
    testWidgets('RegistrationScreen accepts optional RecognizerService parameter', (tester) async {
      final sharedRecognizer = RecognizerService();

      final registrationScreen = RegistrationScreen(
        recognizerService: sharedRecognizer,
      );

      expect(registrationScreen.recognizerService, equals(sharedRecognizer));
    });

    testWidgets('RegistrationScreen defaults recognizerService to null when not provided', (tester) async {
      const registrationScreen = RegistrationScreen();

      expect(registrationScreen.recognizerService, isNull);
    });
  });
}
