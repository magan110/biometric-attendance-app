import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../core/constants/app_constants.dart';
import '../core/utils/math_utils.dart';
import '../data/models/user_model.dart';

class MatchResult {
  final UserModel? user;
  final double distance;
  final bool isMatch;

  MatchResult({
    required this.user,
    required this.distance,
    required this.isMatch,
  });

  /// Approximate confidence percentage derived from cosine distance
  double get confidencePercentage {
    if (distance >= 0.8) return 0.0;
    // Normalized mapping: 0.0 -> 100%, 0.20 -> 75%, 0.40 -> 50%, >= 0.8 -> 0%
    final conf = (1.0 - (distance / 0.8)) * 100;
    return conf.clamp(0.0, 100.0);
  }
}

class RecognizerService {
  Interpreter? _interpreter;
  int _inputWidth = AppConstants.defaultModelInputSize;
  int _inputHeight = AppConstants.defaultModelInputSize;
  int _outputDimension = AppConstants.embeddingDim;

  // Pre-allocated inference buffers to eliminate per-frame GC churn
  List<List<List<List<double>>>>? _inputBuffer;
  List<List<double>>? _outputBuffer;

  int get inputWidth => _inputWidth;
  int get inputHeight => _inputHeight;
  int get outputDimension => _outputDimension;
  bool get isInitialized => _interpreter != null;

  /// Loads the TFLite model from assets with multi-threading (4 threads)
  Future<void> initialize() async {
    final options = InterpreterOptions()..threads = 4;

    try {
      // Primary attempt: facenet_512.tflite present in assets
      _interpreter = await Interpreter.fromAsset(
        AppConstants.faceNet512ModelPath,
        options: options,
      );
      debugPrint('Loaded model from ${AppConstants.faceNet512ModelPath}');
    } catch (_) {
      try {
        // Fallback attempt: mobilefacenet.tflite
        _interpreter = await Interpreter.fromAsset(
          AppConstants.mobileFaceNetModelPath,
          options: options,
        );
        debugPrint('Loaded model from ${AppConstants.mobileFaceNetModelPath}');
      } catch (e) {
        throw Exception(
          'Failed to load TFLite face recognition model from assets. Error: $e',
        );
      }
    }

    // Inspect tensor shape dynamically
    final inputTensors = _interpreter!.getInputTensors();
    if (inputTensors.isNotEmpty) {
      final shape = inputTensors[0].shape;
      if (shape.length >= 4) {
        _inputHeight = shape[1];
        _inputWidth = shape[2];
      }
    }

    final outputTensors = _interpreter!.getOutputTensors();
    if (outputTensors.isNotEmpty) {
      final shape = outputTensors[0].shape;
      _outputDimension = shape.last;
    }

    // Pre-allocate contiguous tensor buffers for zero-allocation prediction loop
    _inputBuffer = List.generate(
      1,
      (_) => List.generate(
        _inputHeight,
        (_) => List.generate(_inputWidth, (_) => [0.0, 0.0, 0.0]),
      ),
    );
    _outputBuffer = List.generate(
      1,
      (_) => List<double>.filled(_outputDimension, 0.0),
    );

    debugPrint(
      'TFLite Interpreter initialized. Input: [1, $_inputHeight, $_inputWidth, 3], Output: [1, $_outputDimension]',
    );
  }

  /// Extracts 512-D L2-normalized embedding vector from a cropped face image
  List<double> predict(img.Image croppedFace) {
    if (_interpreter == null) {
      throw StateError('RecognizerService is not initialized. Call initialize() first.');
    }

    // Ensure image matches model's required input resolution
    img.Image modelInput = croppedFace;
    if (croppedFace.width != _inputWidth || croppedFace.height != _inputHeight) {
      modelInput = img.copyResize(
        croppedFace,
        width: _inputWidth,
        height: _inputHeight,
      );
    }

    // Reuse pre-allocated buffer if dimensions match
    if (_inputBuffer != null &&
        _inputBuffer![0].length == _inputHeight &&
        _inputBuffer![0][0].length == _inputWidth &&
        _outputBuffer != null) {
      final rowBuffer = _inputBuffer![0];
      for (int y = 0; y < _inputHeight; y++) {
        final row = rowBuffer[y];
        for (int x = 0; x < _inputWidth; x++) {
          final pixel = modelInput.getPixel(x, y);
          final ch = row[x];
          ch[0] = (pixel.r - 127.5) / 128.0;
          ch[1] = (pixel.g - 127.5) / 128.0;
          ch[2] = (pixel.b - 127.5) / 128.0;
        }
      }

      _interpreter!.run(_inputBuffer!, _outputBuffer!);
      return MathUtils.l2Normalize(_outputBuffer![0]);
    }

    // Fallback dynamic allocation
    final input = List.generate(
      1,
      (_) => List.generate(
        _inputHeight,
        (y) => List.generate(_inputWidth, (x) {
          final pixel = modelInput.getPixel(x, y);
          return [
            (pixel.r - 127.5) / 128.0,
            (pixel.g - 127.5) / 128.0,
            (pixel.b - 127.5) / 128.0,
          ];
        }),
      ),
    );

    final output = List.generate(
      1,
      (_) => List<double>.filled(_outputDimension, 0.0),
    );

    _interpreter!.run(input, output);
    return MathUtils.l2Normalize(output[0]);
  }

  /// 1:Many search: Compares face embedding with all registered users using Fast Cosine Distance
  /// Evaluates centroid embedding and all enrolled multi-view templates to minimize False Rejections.
  /// Optionally accepts an [alternateEmbedding] (e.g. photometrically normalized) for shadow/glare invariance.
  MatchResult findBestMatch(
    List<double> currentEmbedding,
    List<UserModel> registeredUsers, {
    List<double>? alternateEmbedding,
  }) {
    if (registeredUsers.isEmpty) {
      return MatchResult(user: null, distance: 2.0, isMatch: false);
    }

    UserModel? bestMatchUser;
    double minDistance = double.infinity;

    for (final user in registeredUsers) {
      // 1. Compare against primary/centroid embedding
      double userMinDistance = MathUtils.fastCosineDistance(
        currentEmbedding,
        user.embedding,
      );

      if (alternateEmbedding != null && alternateEmbedding.isNotEmpty) {
        final altDist = MathUtils.fastCosineDistance(
          alternateEmbedding,
          user.embedding,
        );
        if (altDist < userMinDistance) {
          userMinDistance = altDist;
        }
      }

      // 2. Compare against multi-view templates if enrolled
      if (user.templates != null && user.templates!.isNotEmpty) {
        for (final template in user.templates!) {
          final dist = MathUtils.fastCosineDistance(
            currentEmbedding,
            template,
          );
          if (dist < userMinDistance) {
            userMinDistance = dist;
          }

          if (alternateEmbedding != null && alternateEmbedding.isNotEmpty) {
            final altDist = MathUtils.fastCosineDistance(
              alternateEmbedding,
              template,
            );
            if (altDist < userMinDistance) {
              userMinDistance = altDist;
            }
          }
        }
      }

      if (userMinDistance < minDistance) {
        minDistance = userMinDistance;
        bestMatchUser = user;
      }
    }

    final isMatch = minDistance <= AppConstants.matchThreshold;

    return MatchResult(
      user: isMatch ? bestMatchUser : null,
      distance: minDistance,
      isMatch: isMatch,
    );
  }

  /// Closes the TFLite interpreter and releases resources
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _inputBuffer = null;
    _outputBuffer = null;
  }
}
