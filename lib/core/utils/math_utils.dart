import 'dart:math';
import 'package:image/image.dart' as img;

class MathUtils {
  /// Calculates the Cosine Distance between two embedding vectors.
  /// Formula: 1.0 - (dotProduct / (normA * normB))
  /// Returns 0.0 for identical vectors, 1.0 for orthogonal vectors, and 2.0 for opposite vectors.
  static double cosineDistance(List<double> v1, List<double> v2) {
    if (v1.length != v2.length || v1.isEmpty) return 2.0;

    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;

    for (int i = 0; i < v1.length; i++) {
      dotProduct += v1[i] * v2[i];
      normA += v1[i] * v1[i];
      normB += v2[i] * v2[i];
    }

    if (normA <= 0.0 || normB <= 0.0) return 2.0;

    final double denominator = sqrt(normA) * sqrt(normB);
    final double similarity = (dotProduct / denominator).clamp(-1.0, 1.0);
    return 1.0 - similarity;
  }

  /// Calculates Euclidean distance between two vectors as an alternative metric.
  static double euclideanDistance(List<double> v1, List<double> v2) {
    if (v1.length != v2.length || v1.isEmpty) return double.infinity;
    double sum = 0.0;
    for (int i = 0; i < v1.length; i++) {
      final double diff = v1[i] - v2[i];
      sum += diff * diff;
    }
    return sqrt(sum);
  }

  /// L2 normalizes a vector to unit length.
  static List<double> l2Normalize(List<double> vector) {
    if (vector.isEmpty) return vector;
    double sum = 0.0;
    for (final double val in vector) {
      sum += val * val;
    }
    final double norm = sqrt(sum);
    if (norm == 0.0) return List<double>.from(vector);
    return vector.map((val) => val / norm).toList();
  }

  /// Ultra-fast cosine distance for pre-normalized vectors (bypasses sqrt calculations).
  /// For unit vectors, Cosine Distance = 1.0 - dotProduct(v1, v2).
  static double fastCosineDistance(List<double> v1, List<double> v2) {
    if (v1.length != v2.length || v1.isEmpty) return 2.0;
    double dot = 0.0;
    for (int i = 0; i < v1.length; i++) {
      dot += v1[i] * v2[i];
    }
    final similarity = dot.clamp(-1.0, 1.0);
    return 1.0 - similarity;
  }

  /// Calculates the inner dot product of two vectors.
  static double dotProduct(List<double> v1, List<double> v2) {
    if (v1.length != v2.length || v1.isEmpty) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < v1.length; i++) {
      sum += v1[i] * v2[i];
    }
    return sum;
  }

  /// Computes the L2-normalized mean centroid of multiple embedding vectors.
  /// Used for multi-template enrollment to eliminate false rejection across angles.
  static List<double> computeCentroid(List<List<double>> vectors) {
    if (vectors.isEmpty) return [];
    if (vectors.length == 1) return l2Normalize(vectors.first);

    final int dim = vectors.first.length;
    final List<double> sumVector = List<double>.filled(dim, 0.0);

    int validCount = 0;
    for (final v in vectors) {
      if (v.length == dim) {
        validCount++;
        for (int i = 0; i < dim; i++) {
          sumVector[i] += v[i];
        }
      }
    }

    if (validCount == 0) return [];

    for (int i = 0; i < dim; i++) {
      sumVector[i] /= validCount;
    }

    return l2Normalize(sumVector);
  }

  /// Computes arithmetic mean of a collection.
  static double mean(Iterable<num> values) {
    if (values.isEmpty) return 0.0;
    double sum = 0.0;
    int count = 0;
    for (final v in values) {
      sum += v;
      count++;
    }
    return count > 0 ? sum / count : 0.0;
  }

  /// Computes population variance of a collection.
  static double variance(Iterable<num> values) {
    if (values.isEmpty) return 0.0;
    final m = mean(values);
    double sumSquares = 0.0;
    int count = 0;
    for (final v in values) {
      final diff = v - m;
      sumSquares += diff * diff;
      count++;
    }
    return count > 0 ? sumSquares / count : 0.0;
  }

  /// Applies adaptive contrast stretching / photometric normalization
  /// to suppress extreme shadow, harsh backlighting, and dynamic lighting variance.
  /// Uses per-channel dynamic range expansion to preserve natural skin hues.
  static img.Image adaptivePhotometricNormalize(img.Image image) {
    final int width = image.width;
    final int height = image.height;
    if (width == 0 || height == 0) return image;

    int minR = 255, maxR = 0;
    int minG = 255, maxG = 0;
    int minB = 255, maxB = 0;

    // 1. Find dynamic range across each individual RGB channel
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixel(x, y);
        final pr = pixel.r.toInt();
        final pg = pixel.g.toInt();
        final pb = pixel.b.toInt();
        if (pr < minR) minR = pr;
        if (pr > maxR) maxR = pr;
        if (pg < minG) minG = pg;
        if (pg > maxG) maxG = pg;
        if (pb < minB) minB = pb;
        if (pb > maxB) maxB = pb;
      }
    }

    // If dynamic range is already low or degenerate across all channels, return original
    if ((maxR - minR < 15) && (maxG - minG < 15) && (maxB - minB < 15)) {
      return image;
    }

    final double scaleR = maxR > minR ? 255.0 / (maxR - minR) : 1.0;
    final double scaleG = maxG > minG ? 255.0 / (maxG - minG) : 1.0;
    final double scaleB = maxB > minB ? 255.0 / (maxB - minB) : 1.0;

    final img.Image result = img.Image(width: width, height: height);

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final pixel = image.getPixel(x, y);
        final r = maxR > minR
            ? ((pixel.r - minR) * scaleR).round().clamp(0, 255)
            : pixel.r.toInt();
        final g = maxG > minG
            ? ((pixel.g - minG) * scaleG).round().clamp(0, 255)
            : pixel.g.toInt();
        final b = maxB > minB
            ? ((pixel.b - minB) * scaleB).round().clamp(0, 255)
            : pixel.b.toInt();
        result.setPixelRgb(x, y, r, g, b);
      }
    }

    return result;
  }
}

