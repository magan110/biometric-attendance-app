import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class FaceDetectorService {
  late FaceDetector _faceDetector;

  FaceDetectorService() {
    _initializeDetector();
  }

  void _initializeDetector() {
    final options = FaceDetectorOptions(
      performanceMode: FaceDetectorMode.fast,
      enableLandmarks: false,
      enableContours: false,
      enableClassification: true, // Enables eye open and smiling probabilities for liveness
      enableTracking: true, // Enables temporal tracking across frames for passive PAD
      minFaceSize: 0.1, // Improved sensitivity for diverse distances
    );
    _faceDetector = FaceDetector(options: options);
  }

  /// Processes an InputImage and detects faces in the frame
  Future<List<Face>> detectFaces(InputImage inputImage) async {
    return await _faceDetector.processImage(inputImage);
  }

  /// Closes and releases detector resources
  void dispose() {
    _faceDetector.close();
  }
}
