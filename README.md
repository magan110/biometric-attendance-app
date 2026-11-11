# Face Attendance

A Flutter prototype for camera-based face attendance. It combines face detection, face quality checks, blink and passive liveness signals, FaceNet embeddings, and local SQLite enrollment.

> **Prototype notice:** This project has not been independently validated for accuracy, spoof resistance, demographic performance, privacy compliance, or production use. Do not use it as the sole basis for employment, access, payroll, or other consequential decisions. Obtain informed consent and provide a non-biometric alternative.

## What it includes

- Live camera face detection with Google ML Kit
- Face quality checks and blink-based liveness flow
- On-device TFLite face embedding and matching
- Local employee enrollment and embedding storage with SQLite
- Attendance cooldowns and camera lifecycle handling

The app stores names and face embeddings locally on the device. Treat embeddings as sensitive biometric data. The app currently does not provide a documented retention policy, export/deletion workflow for all data, or a privacy notice suitable for deployment.

## Requirements

- Flutter SDK compatible with Dart `^3.13.3`
- A supported device with a camera (Android/iOS; other Flutter targets may need plugin-specific setup)

## Run locally

```sh
flutter pub get
flutter run
```

The face recognition model is expected at `assets/models/facenet_512.tflite`. Verify the model's source, license, and redistribution terms before sharing or distributing this repository. No project license is declared yet; all rights are reserved by default until the project owner chooses and adds a license.

## Project layout

```text
lib/
  core/                 Shared constants and image/math utilities
  data/                 User model and local SQLite database
  ml/                   Detection, recognition, quality, and liveness
  presentation/         Attendance, camera, and registration screens
assets/models/          TFLite face recognition model
```

## Privacy and responsible use

Face embeddings are derived from biometric data and can remain sensitive even when they are not raw photographs. Before any real deployment, document the legal basis and consent process, retention and deletion rules, security controls, access limits, accuracy evaluation, fallback attendance process, and incident response. Camera frames and embeddings should be handled only for a clearly disclosed purpose.

## Marketing copy

See [`docs/MARKETING.md`](docs/MARKETING.md) for a short project description and launch post draft.

## License

No license has been selected. Do not assume the code or bundled model may be reused or redistributed. Add an appropriate license only after confirming rights to both the application code and model asset.
