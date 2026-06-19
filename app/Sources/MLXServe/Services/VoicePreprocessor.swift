import Foundation
import SwiftOGG

/// Decodes a downloaded Telegram audio attachment into the two shapes the
/// bridge needs:
///   1. `pcm` — little-endian float32 mono 16 kHz PCM, the format the Gemma 4
///      unified audio embedder expects (same as a chat-attached clip).
///   2. `fileURL` — an on-disk, AVFoundation-readable file for on-device
///      transcription when the model can't hear audio directly.
///
/// Telegram voice notes are always **Ogg/Opus**, which AVFoundation can't read,
/// so those go through `SwiftOGG` (libopus + libogg) to an intermediate `.m4a`.
/// Everything else (mp3/m4a audio files) is already AVFoundation-readable. Both
/// paths converge on the existing `AudioPreprocessor.preprocess(url:)` for the
/// resample/downmix to 16 kHz mono — no audio plumbing duplicated here.
enum VoicePreprocessor {
    struct Decoded {
        /// float32-LE 16 kHz mono PCM.
        let pcm: Data
        /// AVFoundation-readable file on disk; the caller MUST delete it when done.
        let fileURL: URL
    }

    /// Decode `data` (the raw downloaded bytes). `oggOpus` selects the SwiftOGG
    /// path; `sourceExtension` names the temp file on the non-Opus path so
    /// AVFoundation gets a sensible hint. Returns nil if decoding fails.
    static func decode(_ data: Data, oggOpus: Bool, sourceExtension: String = "m4a") -> Decoded? {
        let fm = FileManager.default
        let stem = fm.temporaryDirectory.appendingPathComponent("tg-audio-\(UUID().uuidString)")
        let decodableURL: URL

        if oggOpus {
            let oga = stem.appendingPathExtension("oga")
            let m4a = stem.appendingPathExtension("m4a")
            do {
                try data.write(to: oga)
                try OGGConverter.convertOpusOGGToM4aFile(src: oga, dest: m4a)
            } catch {
                try? fm.removeItem(at: oga)
                try? fm.removeItem(at: m4a)
                return nil
            }
            try? fm.removeItem(at: oga)
            decodableURL = m4a
        } else {
            let f = stem.appendingPathExtension(sourceExtension.isEmpty ? "m4a" : sourceExtension)
            do { try data.write(to: f) } catch { return nil }
            decodableURL = f
        }

        guard let pcm = AudioPreprocessor.preprocess(url: decodableURL) else {
            try? fm.removeItem(at: decodableURL)
            return nil
        }
        return Decoded(pcm: pcm, fileURL: decodableURL)
    }
}
