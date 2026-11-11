import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../../core/constants/app_constants.dart';
import '../../data/local_db/database_helper.dart';
import '../../data/models/user_model.dart';
import '../../ml/biometrics_pipeline.dart';
import '../../ml/liveness_detector.dart';
import '../camera_view/camera_view.dart';
import '../registration/registration_screen.dart';

enum AttendanceStatus {
  scanning,
  qualityWarning,
  waitingForBlink,
  processing,
  matchSuccess,
  matchFailed,
  cooldownActive,
  spoofDetected,
}

class AttendanceScreen extends StatefulWidget {
  const AttendanceScreen({super.key});

  @override
  State<AttendanceScreen> createState() => _AttendanceScreenState();
}

class _AttendanceScreenState extends State<AttendanceScreen> {
  final BiometricsPipeline _biometricsPipeline = BiometricsPipeline();
  final DatabaseHelper _dbHelper = DatabaseHelper.instance;

  List<UserModel> _registeredUsers = [];
  bool _isLoading = true;
  bool _isFrameWorkerBusy = false;

  AttendanceStatus _status = AttendanceStatus.scanning;
  BiometricsFrameResult? _lastResult;
  Timer? _autoResetTimer;

  String? _attendanceTimestamp;
  int _detectedFacesCount = 0;
  int? _lastPipelineLatencyMs;
  bool _isCameraActive = true;
  int _cameraKeyCounter = 0;

  @override
  void initState() {
    super.initState();
    _initializeServices();
  }

  Future<void> _initializeServices() async {
    try {
      await _biometricsPipeline.recognizerService.initialize();
      await _loadRegisteredUsers();
    } catch (e) {
      debugPrint('Initialization error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Model load warning: $e'),
            backgroundColor: Colors.orange.shade800,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  Future<void> _loadRegisteredUsers() async {
    final users = await _dbHelper.getAllUsers();
    if (mounted) {
      setState(() {
        _registeredUsers = users;
      });
    }
  }

  Future<void> _onFrameProcessed(
    CameraImage image,
    List<Face> faces,
    CameraDescription camera,
  ) async {
    if (!mounted ||
        _isLoading ||
        _isFrameWorkerBusy ||
        _status == AttendanceStatus.matchSuccess ||
        _status == AttendanceStatus.matchFailed ||
        _status == AttendanceStatus.spoofDetected) {
      return;
    }

    _isFrameWorkerBusy = true;
    _detectedFacesCount = faces.length;

    try {
      final result = await _biometricsPipeline.processFrame(
        cameraImage: image,
        faces: faces,
        camera: camera,
        enrolledUsers: _registeredUsers,
      );

      if (!mounted) return;

      _lastPipelineLatencyMs = result.totalLatencyMs;
      _lastResult = result;

      switch (result.status) {
        case BiometricsStatus.idle:
          if (_status != AttendanceStatus.scanning) {
            setState(() {
              _status = AttendanceStatus.scanning;
            });
          }
          break;

        case BiometricsStatus.multipleFaces:
          if (_status != AttendanceStatus.scanning &&
              _status != AttendanceStatus.cooldownActive) {
            setState(() {
              _status = AttendanceStatus.scanning;
            });
          }
          break;

        case BiometricsStatus.qualityGateFailed:
          if (_status != AttendanceStatus.qualityWarning &&
              _status != AttendanceStatus.cooldownActive) {
            setState(() {
              _status = AttendanceStatus.qualityWarning;
            });
          }
          break;

        case BiometricsStatus.waitingForBlink:
          if (_status != AttendanceStatus.cooldownActive) {
            setState(() {
              _status = AttendanceStatus.waitingForBlink;
            });
          }
          break;

        case BiometricsStatus.spoofDetected:
          if (_status != AttendanceStatus.spoofDetected) {
            HapticFeedback.heavyImpact();
            setState(() {
              _status = AttendanceStatus.spoofDetected;
            });
            _autoResetTimer?.cancel();
            _autoResetTimer = Timer(const Duration(milliseconds: 3500), () {
              _biometricsPipeline.reset();
              _resetToScanning();
            });
          }
          break;

        case BiometricsStatus.recognized:
          if (result.isAttendanceMarked) {
            HapticFeedback.mediumImpact();
            final now = DateTime.now();
            final timeStr =
                '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}:${now.second.toString().padLeft(2, '0')}';

            setState(() {
              _status = AttendanceStatus.matchSuccess;
              _attendanceTimestamp = timeStr;
            });

            _autoResetTimer?.cancel();
            _autoResetTimer = Timer(
              AppConstants.attendanceSuccessDisplayDuration,
              _resetToScanning,
            );
          }
          break;

        case BiometricsStatus.cooldownActive:
          if (_status != AttendanceStatus.cooldownActive) {
            setState(() {
              _status = AttendanceStatus.cooldownActive;
            });
          }
          break;

        case BiometricsStatus.notEnrolled:
          if (_status != AttendanceStatus.matchFailed) {
            HapticFeedback.lightImpact();
            setState(() {
              _status = AttendanceStatus.matchFailed;
            });
            _autoResetTimer?.cancel();
            _autoResetTimer =
                Timer(const Duration(milliseconds: 3200), _resetToScanning);
          }
          break;
      }
    } catch (e) {
      debugPrint('Biometrics pipeline frame processing error: $e');
    } finally {
      _isFrameWorkerBusy = false;
    }
  }

  void _resetToScanning() {
    _autoResetTimer?.cancel();
    _biometricsPipeline.reset();
    _isFrameWorkerBusy = false;
    if (mounted) {
      setState(() {
        _status = AttendanceStatus.scanning;
        _lastResult = null;
        _attendanceTimestamp = null;
      });
    }
  }

  Future<void> _navigateToRegistration() async {
    _resetToScanning();
    // 1. Deactivate camera on attendance screen to cleanly release hardware resources
    setState(() {
      _isCameraActive = false;
    });

    // 2. Explicitly await full camera teardown before opening registration
    await CameraView.releaseActiveHardware();
    if (!mounted) return;

    // 3. Open registration screen, sharing already-initialized RecognizerService
    final result = await Navigator.push<bool>(
      context,
      MaterialPageRoute(
        builder: (context) => RegistrationScreen(
          recognizerService: _biometricsPipeline.recognizerService,
        ),
      ),
    );

    // 4. Reload enrolled users if registration was successful
    if (result == true) {
      await _loadRegisteredUsers();
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Employee registered successfully!'),
            backgroundColor: Colors.green,
            duration: Duration(seconds: 2),
          ),
        );
      }
    }

    // 5. Ensure any camera hardware from registration route is completely released before mounting Attendance camera
    if (mounted) {
      await CameraView.releaseActiveHardware();
      if (mounted) {
        _resetToScanning();
        setState(() {
          _cameraKeyCounter++;
          _isCameraActive = true;
        });
      }
    }
  }

  @override
  void dispose() {
    _autoResetTimer?.cancel();
    _biometricsPipeline.recognizerService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text(
          'Face Attendance',
          style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.5),
        ),
        elevation: 0,
        backgroundColor: Colors.black.withValues(alpha: 0.65),
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            tooltip: 'Register New Employee',
            icon: const Icon(Icons.person_add_alt_1),
            onPressed: _navigateToRegistration,
          ),
        ],
      ),
      body: _isLoading
          ? const Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(color: Colors.cyanAccent),
                  SizedBox(height: 16),
                  Text(
                    'Initializing Modern CV Biometrics Engine...',
                    style: TextStyle(color: Colors.white70),
                  ),
                ],
              ),
            )
          : Stack(
              children: [
                // Live camera preview with modernized FacePainter
                if (_isCameraActive)
                  CameraView(
                    key: ValueKey('attendance_cam_$_cameraKeyCounter'),
                    onFrameProcessed: _onFrameProcessed,
                    customLabel: _status == AttendanceStatus.matchSuccess
                        ? '${_lastResult?.matchResult?.user?.name ?? 'Employee'} ✓'
                        : (_status == AttendanceStatus.spoofDetected
                            ? 'Spoof Blocked ⚠️'
                            : (_status == AttendanceStatus.qualityWarning
                                ? _lastResult?.statusMessage
                                : (_status == AttendanceStatus.waitingForBlink
                                    ? _lastResult?.statusMessage
                                    : (_status == AttendanceStatus.cooldownActive
                                        ? _lastResult?.statusMessage
                                        : 'Blink your eyes to check in')))),
                    holdProgress: _lastResult?.blinkResult?.progress ?? 0.0,
                    isMatch: _status == AttendanceStatus.matchSuccess,
                    isSpoofSuspected: _status == AttendanceStatus.spoofDetected,
                    isQualityWarning: _status == AttendanceStatus.qualityWarning,
                    isCooldownActive: _status == AttendanceStatus.cooldownActive,
                    latencyMs: _lastPipelineLatencyMs,
                    isPaused: false,
                  )
                else
                  const Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(color: Colors.cyanAccent),
                        SizedBox(height: 16),
                        Text(
                          'Starting Camera Preview...',
                          style: TextStyle(color: Colors.white70),
                        ),
                      ],
                    ),
                  ),

                // Top Enrolled Employees Badge
                Positioned(
                  top: 16,
                  left: 16,
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 14,
                      vertical: 6,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.black87,
                      borderRadius: BorderRadius.circular(20),
                      border: Border.all(color: Colors.white24),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(
                          Icons.people_alt_outlined,
                          color: Colors.white70,
                          size: 16,
                        ),
                        const SizedBox(width: 6),
                        Text(
                          '${_registeredUsers.length} Enrolled',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 13,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                // Blink Liveness & Latency Diagnostics Badges
                Positioned(
                  top: 16,
                  right: 16,
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      if (_lastPipelineLatencyMs != null)
                        Container(
                          margin: const EdgeInsets.only(right: 8),
                          padding: const EdgeInsets.symmetric(
                            horizontal: 10,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: Colors.black87,
                            borderRadius: BorderRadius.circular(20),
                            border: Border.all(
                              color: _lastPipelineLatencyMs! <= 100
                                  ? Colors.cyanAccent.withValues(alpha: 0.5)
                                  : Colors.orangeAccent.withValues(alpha: 0.5),
                            ),
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                Icons.bolt_rounded,
                                color: _lastPipelineLatencyMs! <= 100
                                    ? Colors.cyanAccent
                                    : Colors.orangeAccent,
                                size: 14,
                              ),
                              const SizedBox(width: 3),
                              Text(
                                '${_lastPipelineLatencyMs}ms',
                                style: TextStyle(
                                  color: _lastPipelineLatencyMs! <= 100
                                      ? Colors.cyanAccent
                                      : Colors.orangeAccent,
                                  fontSize: 11,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                        ),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 12,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xFF1B5E20).withValues(alpha: 0.85),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: Colors.greenAccent.withValues(alpha: 0.4),
                          ),
                        ),
                        child: const Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.remove_red_eye_rounded,
                              color: Colors.greenAccent,
                              size: 14,
                            ),
                            SizedBox(width: 5),
                            Text(
                              'Blink Liveness',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 11,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),

                // Interactive Bottom Status & Confirmation Cards
                Positioned(
                  bottom: 24,
                  left: 16,
                  right: 16,
                  child: _buildBottomStatusCard(),
                ),
              ],
            ),
    );
  }

  Widget _buildBottomStatusCard() {
    switch (_status) {
      case AttendanceStatus.scanning:
        if (_detectedFacesCount > 1) {
          return _buildCard(
            color: const Color(0xFFD32F2F),
            icon: Icons.group_off_rounded,
            title: 'Multiple Faces in Frame!',
            subtitle: 'Only 1 person allowed at a time for biometric security.',
          );
        }
        return _buildCard(
          color: Colors.black87,
          icon: Icons.face_retouching_natural_rounded,
          iconColor: Colors.cyanAccent,
          title: 'Blink your eyes to check in 👁️',
          subtitle:
              'Position face in camera frame and blink eyes to check in.',
        );

      case AttendanceStatus.waitingForBlink:
        final blink = _lastResult?.blinkResult;
        final phase = blink?.phase ?? BlinkPhase.waitingForOpen;
        final progress = blink?.progress ?? 0.35;
        final isClosed = phase == BlinkPhase.eyesClosed;

        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
          decoration: BoxDecoration(
            color: const Color(0xFF003840).withValues(alpha: 0.96),
            borderRadius: BorderRadius.circular(22),
            border: Border.all(
              color: isClosed
                  ? Colors.greenAccent.withValues(alpha: 0.6)
                  : Colors.cyanAccent.withValues(alpha: 0.4),
            ),
            boxShadow: [
              BoxShadow(
                color: (isClosed ? Colors.greenAccent : Colors.cyanAccent)
                    .withValues(alpha: 0.25),
                blurRadius: 18,
              ),
            ],
          ),
          child: Row(
            children: [
              CircleAvatar(
                backgroundColor: (isClosed ? Colors.greenAccent : Colors.cyanAccent)
                    .withValues(alpha: 0.2),
                radius: 26,
                child: Icon(
                  isClosed
                      ? Icons.visibility_off_rounded
                      : Icons.visibility_rounded,
                  color: isClosed ? Colors.greenAccent : Colors.cyanAccent,
                  size: 28,
                ),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      isClosed
                          ? 'Blink Registered! 👁️'
                          : 'Blink your eyes to check in',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _lastResult?.statusMessage ??
                          'Blink your eyes to check in',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12.5,
                      ),
                    ),
                    const SizedBox(height: 8),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: progress,
                        backgroundColor: Colors.white12,
                        valueColor: AlwaysStoppedAnimation<Color>(
                          Color.lerp(
                            Colors.cyanAccent,
                            Colors.greenAccent,
                            progress,
                          )!,
                        ),
                        minHeight: 4,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );

      case AttendanceStatus.qualityWarning:
        return _buildCard(
          color: const Color(0xFFE65100),
          icon: Icons.warning_amber_rounded,
          iconColor: Colors.white,
          title: 'Adjust Face Position',
          subtitle:
              _lastResult?.statusMessage ?? 'Align face towards camera and hold steady.',
        );

      case AttendanceStatus.spoofDetected:
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
          decoration: BoxDecoration(
            color: const Color(0xFFB71C1C).withValues(alpha: 0.98),
            borderRadius: BorderRadius.circular(22),
            boxShadow: [
              BoxShadow(
                color: Colors.redAccent.withValues(alpha: 0.4),
                blurRadius: 20,
              ),
            ],
          ),
          child: Row(
            children: [
              const CircleAvatar(
                backgroundColor: Color(0xFFFF1744),
                radius: 24,
                child: Icon(Icons.shield_outlined, color: Colors.white, size: 28),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Text(
                      'Presentation Attack Blocked! 🛡️',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _lastResult?.statusMessage ??
                          'Digital screen replay or 2D photo spoofing rejected.',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );

      case AttendanceStatus.processing:
        return _buildCard(
          color: const Color(0xFF006064),
          iconWidget: const SizedBox(
            width: 28,
            height: 28,
            child: CircularProgressIndicator(
              strokeWidth: 3,
              color: Colors.cyanAccent,
            ),
          ),
          title: 'Authenticating Face...',
          subtitle: 'Passive PAD verified. Matching biometric vector.',
        );

      case AttendanceStatus.matchSuccess:
        final match = _lastResult?.matchResult;
        final latency = _lastResult?.totalLatencyMs ?? 0;

        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
          decoration: BoxDecoration(
            color: const Color(0xFF1B5E20).withValues(alpha: 0.96),
            borderRadius: BorderRadius.circular(22),
            boxShadow: [
              BoxShadow(
                color: Colors.greenAccent.withValues(alpha: 0.4),
                blurRadius: 20,
                offset: const Offset(0, 4),
              ),
            ],
          ),
          child: Row(
            children: [
              const CircleAvatar(
                backgroundColor: Color(0xFF00E676),
                radius: 26,
                child:
                    Icon(Icons.check_rounded, color: Colors.black87, size: 32),
              ),
              const SizedBox(width: 16),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Row(
                      children: [
                        const Text(
                          'Attendance Marked! ',
                          style: TextStyle(
                            color: Colors.greenAccent,
                            fontSize: 13,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        if (_attendanceTimestamp != null)
                          Text(
                            _attendanceTimestamp!,
                            style: const TextStyle(
                              color: Colors.white70,
                              fontSize: 12,
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 2),
                    Text(
                      match?.user?.name ?? 'Employee',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 0.5,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Latency: ${latency}ms | Liveness: Verified ✓ | Dist: ${match?.distance.toStringAsFixed(3)}',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 11.5,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        );

      case AttendanceStatus.cooldownActive:
        return _buildCard(
          color: const Color(0xFF004D40),
          icon: Icons.check_circle_outline_rounded,
          iconColor: Colors.tealAccent,
          title: 'Already Checked In',
          subtitle: _lastResult?.statusMessage ??
              'Attendance already recorded. Cooldown active.',
        );

      case AttendanceStatus.matchFailed:
        final match = _lastResult?.matchResult;
        return Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: const Color(0xFFB71C1C).withValues(alpha: 0.96),
            borderRadius: BorderRadius.circular(22),
            boxShadow: [
              BoxShadow(
                color: Colors.redAccent.withValues(alpha: 0.3),
                blurRadius: 18,
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(
                children: [
                  const CircleAvatar(
                    backgroundColor: Color(0xFFFF5252),
                    radius: 22,
                    child: Icon(Icons.close_rounded,
                        color: Colors.white, size: 26),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Face Not Recognized',
                          style: TextStyle(
                            color: Colors.white,
                            fontSize: 17,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        Text(
                          match != null
                              ? 'Closest match distance: ${match.distance.toStringAsFixed(3)} (Threshold: ${AppConstants.matchThreshold})'
                              : 'Please ensure your face is registered in system.',
                          style: const TextStyle(
                            color: Colors.white70,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton(
                    onPressed: _resetToScanning,
                    style: TextButton.styleFrom(foregroundColor: Colors.white70),
                    child: const Text('Try Again'),
                  ),
                  const SizedBox(width: 8),
                  ElevatedButton.icon(
                    onPressed: _navigateToRegistration,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: const Color(0xFFB71C1C),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(12),
                      ),
                    ),
                    icon: const Icon(Icons.person_add_alt_1, size: 18),
                    label: const Text(
                      'Register Now',
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ),
                ],
              ),
            ],
          ),
        );
    }
  }

  Widget _buildCard({
    required Color color,
    IconData? icon,
    Color? iconColor,
    Widget? iconWidget,
    required String title,
    required String subtitle,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.95),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white12),
      ),
      child: Row(
        children: [
          iconWidget ??
              Icon(
                icon,
                color: iconColor ?? Colors.white,
                size: 28,
              ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 15,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: const TextStyle(
                    color: Colors.white70,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
