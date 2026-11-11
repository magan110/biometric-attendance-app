class AppConstants {
  /// Dimension of the face recognition embedding vector
  static const int embeddingDim = 512;

  /// Default input resolution (width and height) expected by MobileFaceNet
  static const int defaultModelInputSize = 112;

  /// Cosine distance threshold for a positive face match.
  /// For FaceNet 512, identical faces have distance 0.10 - 0.25,
  /// while unrelated faces have distance > 0.55.
  /// Threshold 0.40 reliably accepts genuine users while rejecting strangers/posters.
  static const double matchThreshold = 0.40;

  /// Hold duration with 1 stable face before auto-capturing
  static const Duration autoCaptureHoldDuration = Duration(milliseconds: 1000);

  /// Duration to display success confirmation card before resetting for next person
  static const Duration attendanceSuccessDisplayDuration =
      Duration(milliseconds: 2800);

  /// Eye open probability threshold to consider eyes fully open (0.0 to 1.0)
  static const double eyeOpenThreshold = 0.65;

  /// Eye open probability threshold to consider eyes closed during a blink (0.0 to 1.0)
  static const double eyeClosedThreshold = 0.25;

  /// Maximum time a stationary face can remain without blinking before flagging static spoof
  static const Duration staticPhotoTimeout = Duration(milliseconds: 3500);

  /// Primary and fallback TFLite asset paths
  static const String mobileFaceNetModelPath =
      'assets/models/mobilefacenet.tflite';
  static const String faceNet512ModelPath = 'assets/models/facenet_512.tflite';

  // -------------------------------------------------------------
  // Face Quality Assessment (FQA) Thresholds (ISO/IEC 29794-5)
  // -------------------------------------------------------------
  /// Minimum face bounding box dimension in pixels to qualify for recognition
  static const double minFaceResolution = 60.0;

  /// Maximum absolute head yaw angle in degrees (head turned left/right)
  static const double maxPoseYawAngle = 35.0;

  /// Maximum absolute head pitch angle in degrees (head tilted up/down)
  static const double maxPosePitchAngle = 25.0;

  /// Maximum absolute head roll angle in degrees (head tilted sideways)
  static const double maxPoseRollAngle = 20.0;

  /// Minimum Laplacian variance score for blur rejection (motion blur filter)
  static const double minSharpnessScore = 18.0;

  /// Minimum average luminance (0-255) to prevent underexposed/pitch-black frames
  static const double minLuminance = 32.0;

  /// Maximum average luminance (0-255) to prevent washed-out/overexposed glare
  static const double maxLuminance = 230.0;

  /// Composite face quality pass score (0.0 to 1.0)
  static const double minFaceQualityScore = 0.50;

  // -------------------------------------------------------------
  // Passive Presentation Attack Detection (PAD - ISO/IEC 30107-3)
  // -------------------------------------------------------------
  /// Minimum composite passive liveness score required to confirm live human
  static const double passiveLivenessThreshold = 0.65;

  /// High-frequency Moiré pattern ratio threshold above which screen replays are flagged
  static const double moireScreenThreshold = 0.65;

  /// Frame window count used for passive temporal micro-dynamics tracking
  static const int temporalTrackingWindow = 5;

  // -------------------------------------------------------------
  // Walk-Through Frictionless Pipeline Parameters
  // -------------------------------------------------------------
  /// Target total pipeline latency budget in milliseconds (edge device)
  static const int targetPipelineLatencyMs = 100;

  /// Cooldown duration between attendance marks for the same recognized user
  static const Duration walkThroughCooldown = Duration(seconds: 30);
}

