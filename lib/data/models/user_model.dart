import 'dart:convert';

class UserModel {
  final int? id;
  final String name;
  final List<double> embedding;
  /// Optional multi-angle/multi-lighting face templates for bulletproof recognition
  final List<List<double>>? templates;
  final String? registeredAt;

  UserModel({
    this.id,
    required this.name,
    required this.embedding,
    this.templates,
    this.registeredAt,
  });

  /// Convert object to Map for SQLite persistence
  Map<String, dynamic> toMap() {
    final map = <String, dynamic>{
      'id': id,
      'name': name,
      'embedding': jsonEncode(embedding),
      'registered_at': registeredAt ?? DateTime.now().toIso8601String(),
    };
    if (templates != null && templates!.isNotEmpty) {
      map['templates'] = jsonEncode(templates);
    }
    return map;
  }

  /// Factory constructor to reconstruct object from SQLite query result
  factory UserModel.fromMap(Map<String, dynamic> map) {
    final embedding = (jsonDecode(map['embedding'] as String) as List)
        .map((e) => (e as num).toDouble())
        .toList();

    List<List<double>>? parsedTemplates;
    if (map['templates'] != null) {
      try {
        final decoded = jsonDecode(map['templates'] as String);
        if (decoded is List) {
          parsedTemplates = decoded
              .map((item) =>
                  (item as List).map((e) => (e as num).toDouble()).toList())
              .toList();
        }
      } catch (_) {}
    }

    return UserModel(
      id: map['id'] as int?,
      name: map['name'] as String,
      embedding: embedding,
      templates: parsedTemplates,
      registeredAt: map['registered_at'] as String?,
    );
  }
}

