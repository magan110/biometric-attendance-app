import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img;

import '../../core/utils/image_converter.dart';
import '../../core/utils/math_utils.dart';
import '../../data/local_db/database_helper.dart';
import '../../data/models/user_model.dart';
import '../../ml/face_quality_assessment.dart';
import '../../ml/recognizer_service.dart';
import '../camera_view/camera_view.dart';

class RegistrationScreen extends StatefulWidget {
  final RecognizerService? recognizerService;

  const RegistrationScreen({
    super.key,
    this.recognizerService,
  });

  @override
  State<RegistrationScreen> createState() => _RegistrationScreenState();
}

class _RegistrationScreenState extends State<RegistrationScreen> {
  final TextEditingController _nameController = TextEditingController();
  late final RecognizerService _recognizerService;
  late final bool _ownsRecognizerService;
  final DatabaseHelper _dbHelper = DatabaseHelper.instance;
  final FaceQualityAssessment _qualityAssessment = FaceQualityAssessment();

  CameraImage? _latestCameraImage;
  CameraDescription? _latestCamera;
  img.Image? _latestCroppedFace;
  List<Face> _detectedFaces = [];
  FaceQualityResult? _latestQualityResult;

  bool _isModelReady = false;
  bool _isRegistering = false;
  bool _isCameraPaused = false;

  @override
  void initState() {
    super.initState();
    if (widget.recognizerService != null) {
      _recognizerService = widget.recognizerService!;
      _ownsRecognizerService = false;
      _isModelReady = _recognizerService.isInitialized;
      if (!_isModelReady) {
        _initModel();
      }
    } else {
      _recognizerService = RecognizerService();
      _ownsRecognizerService = true;
      _initModel();
    }
  }

  Future<void> _initModel() async {
    try {
      await _recognizerService.initialize();
      if (mounted) {
        setState(() {
          _isModelReady = true;
        });
      }
    } catch (e) {
      debugPrint('Registration model init error: $e');
    }
  }

  void _onFrameProcessed(
    CameraImage image,
    List<Face> faces,
    CameraDescription camera,
  ) {
    if (!mounted || _isRegistering || _isCameraPaused) return;

    FaceQualityResult? quality;
    img.Image? crop;
    if (faces.length == 1) {
      final face = faces.first;
      crop = ImageConverter.cropFaceDirect(
        image,
        face.boundingBox,
        _recognizerService.inputWidth,
        camera.sensorOrientation,
      );
      final bool isRotated = camera.sensorOrientation == 90 ||
          camera.sensorOrientation == 270;
      final Size uprightSize = Size(
        isRotated ? image.height.toDouble() : image.width.toDouble(),
        isRotated ? image.width.toDouble() : image.height.toDouble(),
      );
      quality = _qualityAssessment.assessQuality(
        face: face,
        faceCrop: crop,
        imageSize: uprightSize,
      );
    }

    setState(() {
      _latestCameraImage = image;
      _latestCroppedFace = crop;
      _latestCamera = camera;
      _detectedFaces = faces;
      _latestQualityResult = quality;
    });
  }

  Future<void> _registerUser() async {
    final String name = _nameController.text.trim();
    if (name.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter employee name!'),
          backgroundColor: Colors.redAccent,
        ),
      );
      return;
    }

    final Face? face =
        _detectedFaces.isNotEmpty ? _detectedFaces.first : null;
    final CameraImage? image = _latestCameraImage;
    final CameraDescription? camera = _latestCamera;

    if (_detectedFaces.length != 1 ||
        face == null ||
        image == null ||
        camera == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please ensure exactly one face is clearly visible!'),
          backgroundColor: Colors.orange,
        ),
      );
      return;
    }

    if (_latestQualityResult != null && !_latestQualityResult!.isAcceptable) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            'Cannot enroll: ${_latestQualityResult!.feedbackMessage}',
          ),
          backgroundColor: Colors.orange.shade800,
        ),
      );
      return;
    }

    setState(() {
      _isRegistering = true;
    });

    try {
      // 1. Direct low-latency face crop (prefer cached frame crop)
      final img.Image croppedFace = _latestCroppedFace ??
          ImageConverter.cropFaceDirect(
            image,
            face.boundingBox,
            _recognizerService.inputWidth,
            camera.sensorOrientation,
          );

      // 2. Extract primary 512-D embedding
      final List<double> primaryEmbedding =
          _recognizerService.predict(croppedFace);

      // 3. Generate adaptive contrast-normalized template for lighting invariance
      final img.Image contrastNormalizedFace =
          MathUtils.adaptivePhotometricNormalize(croppedFace);
      final List<double> normalizedEmbedding =
          _recognizerService.predict(contrastNormalizedFace);

      // 4. Compute optimal centroid embedding
      final List<double> centroid = MathUtils.computeCentroid([
        primaryEmbedding,
        normalizedEmbedding,
      ]);

      // 5. Store {name, centroid, multi-view templates} into SQLite
      final user = UserModel(
        name: name,
        embedding: centroid,
        templates: [primaryEmbedding, normalizedEmbedding],
        registeredAt: DateTime.now().toIso8601String(),
      );

      await _dbHelper.insertUser(user);

      if (!mounted) return;

      // 6. Explicitly pause and release camera hardware before navigating back
      setState(() {
        _isCameraPaused = true;
      });

      // Await complete native hardware release before popping route
      await CameraView.releaseActiveHardware();

      // 7. Success confirmation & return to attendance
      if (mounted) {
        Navigator.pop(context, true);
      }
    } catch (e) {
      debugPrint('Registration error: $e');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to register face: $e'),
            backgroundColor: Colors.redAccent,
          ),
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _isRegistering = false;
        });
      }
    }
  }

  Future<void> _handleBack() async {
    if (_isRegistering) return;
    setState(() {
      _isCameraPaused = true;
    });
    await CameraView.releaseActiveHardware();
    if (mounted) {
      Navigator.pop(context, false);
    }
  }

  @override
  void dispose() {
    _nameController.dispose();
    if (_ownsRecognizerService) {
      _recognizerService.dispose();
    }
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final bool isQualityGood =
        _latestQualityResult != null && _latestQualityResult!.isAcceptable;
    final bool canRegister = _isModelReady &&
        !_isRegistering &&
        _detectedFaces.length == 1 &&
        isQualityGood &&
        _nameController.text.trim().isNotEmpty;

    return PopScope(
      canPop: false,
      onPopInvokedWithResult: (didPop, result) async {
        if (didPop) return;
        await _handleBack();
      },
      child: Scaffold(
        backgroundColor: Colors.black,
        appBar: AppBar(
          title: const Text('Enroll Employee Face'),
          backgroundColor: Colors.black.withValues(alpha: 0.65),
          foregroundColor: Colors.white,
          elevation: 0,
          leading: IconButton(
            icon: const Icon(Icons.arrow_back),
            onPressed: _handleBack,
          ),
        ),
        body: Stack(
          children: [
            // Camera Preview with Bounding Box
            CameraView(
              isPaused: _isCameraPaused,
              onFrameProcessed: _onFrameProcessed,
              recognizedName: _nameController.text.isNotEmpty
                  ? _nameController.text.trim()
                  : 'Enrolling Face',
              isMatch: isQualityGood,
              isQualityWarning: _latestQualityResult != null && !isQualityGood,
              customLabel: _latestQualityResult?.feedbackMessage,
            ),

          // Face Status Indicator Chip
          Positioned(
            top: 16,
            left: 16,
            right: 16,
            child: Center(child: _buildFaceStatusChip()),
          ),

          // Bottom Input & Registration Card
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: Container(
              padding: const EdgeInsets.fromLTRB(20, 20, 20, 32),
              decoration: BoxDecoration(
                color: Colors.grey.shade900.withValues(alpha: 0.95),
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(24)),
                boxShadow: const [
                  BoxShadow(
                    color: Colors.black54,
                    blurRadius: 16,
                    offset: Offset(0, -4),
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  TextField(
                    controller: _nameController,
                    style: const TextStyle(color: Colors.white),
                    decoration: InputDecoration(
                      labelText: 'Employee / User Name',
                      labelStyle: const TextStyle(color: Colors.white70),
                      hintText: 'e.g. Rahul Sharma',
                      hintStyle: const TextStyle(color: Colors.white30),
                      prefixIcon:
                          const Icon(Icons.badge, color: Colors.cyanAccent),
                      suffixIcon: _nameController.text.isNotEmpty
                          ? IconButton(
                              icon: const Icon(Icons.clear,
                                  color: Colors.white54),
                              onPressed: () {
                                _nameController.clear();
                                setState(() {});
                              },
                            )
                          : null,
                      filled: true,
                      fillColor: Colors.black38,
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(14),
                        borderSide: const BorderSide(color: Colors.white24),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(14),
                        borderSide: const BorderSide(color: Colors.white24),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(14),
                        borderSide: const BorderSide(color: Colors.cyanAccent),
                      ),
                    ),
                    onChanged: (_) => setState(() {}),
                  ),
                  const SizedBox(height: 16),
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: canRegister
                          ? Colors.cyanAccent.shade700
                          : Colors.grey.shade700,
                      foregroundColor: Colors.white,
                      disabledBackgroundColor: Colors.grey.shade800,
                      disabledForegroundColor: Colors.white38,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                      elevation: canRegister ? 4 : 0,
                    ),
                    onPressed: canRegister ? _registerUser : null,
                    child: _isRegistering
                        ? const Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              SizedBox(
                                width: 20,
                                height: 20,
                                child: CircularProgressIndicator(
                                  strokeWidth: 2,
                                  color: Colors.white,
                                ),
                              ),
                              SizedBox(width: 12),
                              Text('Enrolling Multi-Template Vector...'),
                            ],
                          )
                        : const Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(Icons.camera_alt),
                              SizedBox(width: 8),
                              Text(
                                'Capture & Enroll Face',
                                style: TextStyle(
                                  fontSize: 16,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    ),
  );
}

  Widget _buildFaceStatusChip() {
    if (_detectedFaces.isEmpty) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black87,
          borderRadius: BorderRadius.circular(20),
          border:
              Border.all(color: Colors.amberAccent.withValues(alpha: 0.6)),
        ),
        child: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.face, color: Colors.amberAccent, size: 18),
            SizedBox(width: 8),
            Text(
              'No face detected - Look at camera',
              style: TextStyle(color: Colors.amberAccent, fontSize: 13),
            ),
          ],
        ),
      );
    }

    if (_detectedFaces.length > 1) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black87,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.redAccent.withValues(alpha: 0.6)),
        ),
        child: const Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.group, color: Colors.redAccent, size: 18),
            SizedBox(width: 8),
            Text(
              'Multiple faces! Only 1 person allowed',
              style: TextStyle(color: Colors.redAccent, fontSize: 13),
            ),
          ],
        ),
      );
    }

    if (_latestQualityResult != null && !_latestQualityResult!.isAcceptable) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
        decoration: BoxDecoration(
          color: Colors.black87,
          borderRadius: BorderRadius.circular(20),
          border:
              Border.all(color: Colors.orangeAccent.withValues(alpha: 0.8)),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.warning_amber_rounded,
                color: Colors.orangeAccent, size: 18),
            const SizedBox(width: 8),
            Text(
              _latestQualityResult!.feedbackMessage,
              style: const TextStyle(color: Colors.orangeAccent, fontSize: 13),
            ),
          ],
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(20),
        border:
            Border.all(color: Colors.greenAccent.withValues(alpha: 0.6)),
      ),
      child: const Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.check_circle_outline, color: Colors.greenAccent, size: 18),
          SizedBox(width: 8),
          Text(
            'Quality Optimal ✓ Ready to Enroll',
            style: TextStyle(color: Colors.greenAccent, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
