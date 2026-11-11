import 'dart:async';
import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

import '../../core/utils/image_converter.dart';
import 'face_painter.dart';

class CameraView extends StatefulWidget {
  final FutureOr<void> Function(
    CameraImage image,
    List<Face> faces,
    CameraDescription camera,
  ) onFrameProcessed;
  final String? recognizedName;
  final String? customLabel;
  final double holdProgress;
  final bool isSpoofSuspected;
  final bool isMatch;
  final bool isPaused;
  final bool isQualityWarning;
  final bool isCooldownActive;
  final int? latencyMs;

  const CameraView({
    super.key,
    required this.onFrameProcessed,
    this.recognizedName,
    this.customLabel,
    this.holdProgress = 0.0,
    this.isMatch = false,
    this.isSpoofSuspected = false,
    this.isPaused = false,
    this.isQualityWarning = false,
    this.isCooldownActive = false,
    this.latencyMs,
  });

  static Future<void>? _globalDisposalFuture;
  static _CameraViewState? _activeInstance;

  /// Global safe hardware release: shuts down any active camera controller across the app,
  /// awaits native OS release, and allows Android HAL session state to settle.
  static Future<void> releaseActiveHardware() async {
    final instance = _activeInstance;
    if (instance != null) {
      await instance._disposeCamera();
    }
    if (_globalDisposalFuture != null) {
      try {
        await _globalDisposalFuture;
      } catch (_) {}
    }
    // Brief settling delay for Android OS CameraService HAL
    await Future.delayed(const Duration(milliseconds: 100));
  }

  @override
  State<CameraView> createState() => _CameraViewState();
}

class _CameraViewState extends State<CameraView> with WidgetsBindingObserver {
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  int _cameraIndex = 0;
  bool _isProcessingFrame = false;
  List<Face> _faces = [];
  late FaceDetector _faceDetector;
  String? _errorMessage;
  bool _isDisposed = false;
  bool _isInitializing = false;

  @override
  void initState() {
    super.initState();
    CameraView._activeInstance = this;
    WidgetsBinding.instance.addObserver(this);
    _initDetector();
    if (!widget.isPaused) {
      _initCamera();
    }
  }

  void _initDetector() {
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.fast,
        enableLandmarks: false,
        enableContours: false,
        enableClassification: true,
        enableTracking: true, // Enables temporal tracking across frames for passive PAD
        minFaceSize: 0.1, // High sensitivity for face detection
      ),
    );
  }

  void _onCameraControllerUpdate() {
    if (!mounted || _isDisposed) return;
    final controller = _cameraController;
    if (controller != null && controller.value.hasError) {
      debugPrint('CameraController error: ${controller.value.errorDescription}');
      if (mounted) {
        setState(() {
          _errorMessage =
              'Camera hardware error: ${controller.value.errorDescription}';
        });
      }
    }
  }

  Future<void> _disposeCamera() async {
    final controller = _cameraController;
    _cameraController = null;
    if (controller != null) {
      controller.removeListener(_onCameraControllerUpdate);
      final completer = Completer<void>();
      CameraView._globalDisposalFuture = completer.future;
      try {
        if (controller.value.isStreamingImages) {
          await controller.stopImageStream();
        }
      } catch (e) {
        debugPrint('Camera stopImageStream error during dispose: $e');
      }
      try {
        await controller.dispose();
      } catch (e) {
        debugPrint('Camera controller.dispose error: $e');
      } finally {
        completer.complete();
      }
    }
  }

  Future<void> _initCamera({int retryCount = 0}) async {
    if (_isDisposed || widget.isPaused) return;
    if (_isInitializing) return;
    _isInitializing = true;

    CameraController? tempController;
    try {
      // 1. Await any global in-flight camera teardown
      if (CameraView._globalDisposalFuture != null) {
        try {
          await CameraView._globalDisposalFuture;
        } catch (_) {}
      }

      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        if (mounted) {
          setState(() {
            _errorMessage = 'No camera found on device';
          });
        }
        return;
      }

      // Default to front camera for face attendance
      _cameraIndex = _cameras.indexWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
      );
      if (_cameraIndex == -1) _cameraIndex = 0;

      await _disposeCamera();
      if (_isDisposed || !mounted || widget.isPaused) return;

      final ImageFormatGroup preferredFormat = Platform.isAndroid
          ? ImageFormatGroup.nv21
          : ImageFormatGroup.bgra8888;

      tempController = CameraController(
        _cameras[_cameraIndex],
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: preferredFormat,
      );

      bool initSuccess = false;
      try {
        await tempController.initialize();
        initSuccess = true;
      } catch (e) {
        debugPrint('Camera format $preferredFormat init failed: $e');
        // Always dispose the failed controller immediately to prevent leaking native handles
        await tempController.dispose().catchError((_) {});
        tempController = null;

        if (_isDisposed || !mounted || widget.isPaused) return;

        final errStr = e.toString().toLowerCase();
        final isBusy = errStr.contains('busy') ||
            errStr.contains('in use') ||
            errStr.contains('locked');

        // Only try fallback format if error wasn't camera hardware busy
        if (!isBusy && Platform.isAndroid) {
          tempController = CameraController(
            _cameras[_cameraIndex],
            ResolutionPreset.medium,
            enableAudio: false,
            imageFormatGroup: ImageFormatGroup.yuv420,
          );
          await tempController.initialize();
          initSuccess = true;
        } else {
          rethrow;
        }
      }

      if (!initSuccess || _isDisposed || !mounted || widget.isPaused) {
        await tempController.dispose().catchError((_) {});
        tempController = null;
        return;
      }

      tempController.addListener(_onCameraControllerUpdate);
      _cameraController = tempController;
      tempController = null; // Ownership transferred to _cameraController
      CameraView._activeInstance = this;

      await _cameraController!.startImageStream(_processCameraFrame);

      if (mounted) {
        setState(() {
          _errorMessage = null;
        });
      }
    } catch (e) {
      debugPrint('Camera initialization error (attempt $retryCount): $e');
      if (tempController != null) {
        await tempController.dispose().catchError((_) {});
        tempController = null;
      }

      if (_isDisposed || !mounted) return;

      if (retryCount < 3 && !widget.isPaused) {
        // Exponential backoff: 400ms, 800ms, 1200ms
        await Future.delayed(Duration(milliseconds: 400 * (retryCount + 1)));
        if (!_isDisposed && mounted && !widget.isPaused) {
          _isInitializing = false;
          return _initCamera(retryCount: retryCount + 1);
        }
      }

      if (mounted) {
        setState(() {
          _errorMessage = 'Camera initialization failed: $e';
        });
      }
    } finally {
      _isInitializing = false;
    }
  }

  Future<void> _processCameraFrame(CameraImage image) async {
    if (_isDisposed || _isProcessingFrame || !mounted || widget.isPaused) return;
    _isProcessingFrame = true;

    try {
      final camera = _cameras[_cameraIndex];
      final inputImage = ImageConverter.convertCameraImageToInputImage(
        image,
        camera,
        deviceOrientation: _cameraController?.value.deviceOrientation ??
            DeviceOrientation.portraitUp,
      );

      if (inputImage != null) {
        final faces = await _faceDetector.processImage(inputImage);
        if (_isDisposed || !mounted || widget.isPaused) return;

        setState(() {
          _faces = faces;
          _errorMessage = null;
        });
        await widget.onFrameProcessed(image, faces, camera);
      }
    } catch (e) {
      debugPrint('Face detection frame error: $e');
      if (mounted && _errorMessage == null && !_isDisposed) {
        setState(() {
          _errorMessage = 'Detection error: $e';
        });
      }
    } finally {
      _isProcessingFrame = false;
    }
  }

  @override
  void didUpdateWidget(covariant CameraView oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.isPaused != oldWidget.isPaused) {
      if (widget.isPaused) {
        _disposeCamera();
        if (mounted) {
          setState(() {
            _faces = [];
          });
        }
      } else {
        _initCamera();
      }
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_isDisposed) return;

    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _disposeCamera();
    } else if (state == AppLifecycleState.resumed) {
      if (!widget.isPaused) {
        _initCamera();
      }
    }
  }

  @override
  void dispose() {
    _isDisposed = true;
    if (identical(CameraView._activeInstance, this)) {
      CameraView._activeInstance = null;
    }
    WidgetsBinding.instance.removeObserver(this);
    _disposeCamera();
    _faceDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_errorMessage != null &&
        (_cameraController == null || !_cameraController!.value.isInitialized)) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(
                Icons.videocam_off_outlined,
                color: Colors.orangeAccent,
                size: 48,
              ),
              const SizedBox(height: 16),
              Text(
                _errorMessage!,
                style: const TextStyle(color: Colors.white70, fontSize: 13),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 16),
              ElevatedButton.icon(
                onPressed: () => _initCamera(),
                icon: const Icon(Icons.refresh),
                label: const Text('Restart Camera'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.cyanAccent.shade700,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(color: Colors.white),
            const SizedBox(height: 16),
            Text(
              widget.isPaused ? 'Camera paused' : 'Initializing camera...',
              style: const TextStyle(color: Colors.white70),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    final camera = _cameras[_cameraIndex];
    final previewSize = _cameraController!.value.previewSize!;

    return Stack(
      fit: StackFit.expand,
      children: [
        // Camera Preview with embedded child painter for 1:1 pixel alignment
        Center(
          child: CameraPreview(
            _cameraController!,
            child: CustomPaint(
              painter: FacePainter(
                faces: widget.isPaused ? [] : _faces,
                imageSize: Size(previewSize.height, previewSize.width),
                cameraLensDirection: camera.lensDirection,
                customLabel: widget.customLabel ?? widget.recognizedName,
                holdProgress: widget.holdProgress,
                isMatch: widget.isMatch,
                isSpoofSuspected: widget.isSpoofSuspected,
                isQualityWarning: widget.isQualityWarning,
                isCooldownActive: widget.isCooldownActive,
                latencyMs: widget.latencyMs,
              ),
            ),
          ),
        ),

        // Error banner if any non-fatal exception occurred
        if (_errorMessage != null)
          Positioned(
            top: 40,
            left: 20,
            right: 20,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.red.shade900.withValues(alpha: 0.9),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: Text(
                      _errorMessage!,
                      style: const TextStyle(color: Colors.white, fontSize: 12),
                      textAlign: TextAlign.center,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.refresh, color: Colors.white, size: 18),
                    onPressed: () => _initCamera(),
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }
}
