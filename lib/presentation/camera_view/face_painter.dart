import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FacePainter extends CustomPainter {
  final List<Face> faces;
  final Size imageSize;
  final CameraLensDirection cameraLensDirection;
  final double holdProgress; // 0.0 to 1.0
  final String? customLabel;
  final bool isMatch;
  final bool isSpoofSuspected;
  final bool isQualityWarning;
  final bool isCooldownActive;
  final int? latencyMs;

  FacePainter({
    required this.faces,
    required this.imageSize,
    required this.cameraLensDirection,
    this.holdProgress = 0.0,
    this.customLabel,
    this.isMatch = false,
    this.isSpoofSuspected = false,
    this.isQualityWarning = false,
    this.isCooldownActive = false,
    this.latencyMs,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (faces.isEmpty || imageSize.width == 0 || imageSize.height == 0) return;

    final double scaleX = size.width / imageSize.width;
    final double scaleY = size.height / imageSize.height;
    final bool isMultiple = faces.length > 1;

    for (int i = 0; i < faces.length; i++) {
      final Face face = faces[i];
      double left = face.boundingBox.left * scaleX;
      double top = face.boundingBox.top * scaleY;
      double right = face.boundingBox.right * scaleX;
      double bottom = face.boundingBox.bottom * scaleY;

      // Front camera horizontal mirroring compensation
      if (cameraLensDirection == CameraLensDirection.front) {
        final double originalLeft = left;
        left = size.width - right;
        right = size.width - originalLeft;
      }

      final Rect rect = Rect.fromLTRB(left, top, right, bottom);
      final RRect rrect =
          RRect.fromRectAndRadius(rect, const Radius.circular(16.0));

      Color strokeColor;
      Color fillColor;
      String label;

      if (isMultiple) {
        // Red for multiple faces (guard rule)
        strokeColor = const Color(0xFFFF5252);
        fillColor = const Color(0xFFFF5252).withValues(alpha: 0.12);
        label = 'Multiple Faces (${i + 1})';
      } else if (isSpoofSuspected) {
        // Crimson Red for spoof / replay / print attack
        strokeColor = const Color(0xFFFF1744);
        fillColor = const Color(0xFFFF1744).withValues(alpha: 0.22);
        label = customLabel ?? 'Spoof Blocked ⚠️';
      } else if (isMatch) {
        // Emerald Green for recognized employee
        strokeColor = const Color(0xFF00E676);
        fillColor = const Color(0xFF00E676).withValues(alpha: 0.22);
        label = customLabel ?? 'Verified ✓';
      } else if (isCooldownActive) {
        // Teal for active cooldown (already checked in)
        strokeColor = const Color(0xFF26A69A);
        fillColor = const Color(0xFF26A69A).withValues(alpha: 0.18);
        label = customLabel ?? 'Already Checked In ✓';
      } else if (isQualityWarning) {
        // Amber for FQA warning (motion blur, low light, steep angle)
        strokeColor = const Color(0xFFFFC107);
        fillColor = const Color(0xFFFFC107).withValues(alpha: 0.15);
        label = customLabel ?? 'Adjust Face Angle';
      } else if (holdProgress >= 1.0) {
        // Liveness confirmed
        strokeColor = const Color(0xFF00E676);
        fillColor = const Color(0xFF00E676).withValues(alpha: 0.20);
        label = customLabel ?? 'Liveness Verified ✓';
      } else if (holdProgress > 0.0) {
        // In verification progress
        strokeColor = Color.lerp(
          const Color(0xFF00E5FF),
          const Color(0xFF00E676),
          holdProgress,
        )!;
        fillColor = strokeColor.withValues(alpha: 0.15);
        label = customLabel ?? 'Blink your eyes to check in';
      } else {
        // Default cyan for walk-through biometric scanning
        strokeColor = const Color(0xFF00E5FF);
        fillColor = const Color(0xFF00E5FF).withValues(alpha: 0.08);
        label = customLabel ?? 'Blink your eyes to check in';
      }

      // Fill background
      final Paint bgPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = fillColor;
      canvas.drawRRect(rrect, bgPaint);

      // Stroke bounding box
      final Paint boxPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5
        ..color = strokeColor.withValues(alpha: 0.6);
      canvas.drawRRect(rrect, boxPaint);

      // Draw high-tech HUD corner brackets
      final double bracketLength = (rect.width * 0.18).clamp(16.0, 36.0);
      final Paint bracketPaint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 4.0
        ..strokeCap = StrokeCap.round
        ..color = strokeColor;

      // Top-Left
      canvas.drawLine(
        Offset(left, top + bracketLength),
        Offset(left, top),
        bracketPaint,
      );
      canvas.drawLine(
        Offset(left, top),
        Offset(left + bracketLength, top),
        bracketPaint,
      );

      // Top-Right
      canvas.drawLine(
        Offset(right - bracketLength, top),
        Offset(right, top),
        bracketPaint,
      );
      canvas.drawLine(
        Offset(right, top),
        Offset(right, top + bracketLength),
        bracketPaint,
      );

      // Bottom-Left
      canvas.drawLine(
        Offset(left, bottom - bracketLength),
        Offset(left, bottom),
        bracketPaint,
      );
      canvas.drawLine(
        Offset(left, bottom),
        Offset(left + bracketLength, bottom),
        bracketPaint,
      );

      // Bottom-Right
      canvas.drawLine(
        Offset(right - bracketLength, bottom),
        Offset(right, bottom),
        bracketPaint,
      );
      canvas.drawLine(
        Offset(right, bottom),
        Offset(right, bottom - bracketLength),
        bracketPaint,
      );

      // Draw Top Status Pill
      final String latencySuffix =
          latencyMs != null ? ' (${latencyMs}ms)' : '';
      final TextSpan textSpan = TextSpan(
        text: ' $label$latencySuffix ',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12.5,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.3,
        ),
      );

      final TextPainter textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      final double tagWidth = textPainter.width + 12;
      final double tagHeight = textPainter.height + 6;
      final double maxTagLeft =
          (size.width - tagWidth).clamp(0.0, double.infinity);
      final double maxTagTop =
          (size.height - tagHeight).clamp(0.0, double.infinity);
      final double tagLeft = left.clamp(0.0, maxTagLeft);
      final double tagTop = (top - tagHeight - 8).clamp(0.0, maxTagTop);

      final RRect tagBg = RRect.fromRectAndRadius(
        Rect.fromLTWH(tagLeft, tagTop, tagWidth, tagHeight),
        const Radius.circular(8.0),
      );

      final Paint tagBgPaint = Paint()
        ..style = PaintingStyle.fill
        ..color = isMultiple
            ? const Color(0xFFB71C1C)
            : (isSpoofSuspected
                ? const Color(0xFFD50000)
                : (isMatch
                    ? const Color(0xFF1B5E20)
                    : (isCooldownActive
                        ? const Color(0xFF004D40)
                        : (isQualityWarning
                            ? const Color(0xFFE65100)
                            : const Color(0xFF006064).withValues(alpha: 0.90)))));

      canvas.drawRRect(tagBg, tagBgPaint);
      textPainter.paint(canvas, Offset(tagLeft + 6, tagTop + 3));
    }
  }

  @override
  bool shouldRepaint(FacePainter oldDelegate) {
    return oldDelegate.faces != faces ||
        oldDelegate.holdProgress != holdProgress ||
        oldDelegate.customLabel != customLabel ||
        oldDelegate.isMatch != isMatch ||
        oldDelegate.isCooldownActive != isCooldownActive ||
        oldDelegate.isSpoofSuspected != isSpoofSuspected ||
        oldDelegate.isQualityWarning != isQualityWarning ||
        oldDelegate.latencyMs != latencyMs;
  }
}
