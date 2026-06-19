import XCTest
import SwiftOGG
@testable import MLXCore

/// Pins the one genuinely new capability for Telegram voice: decoding an
/// Ogg/Opus voice note (the only container Telegram sends, and the one
/// AVFoundation can't read) into the 16 kHz mono float32 PCM the model wants —
/// plus the AVFoundation passthrough branch for plain audio files. Fixtures are
/// synthesised at runtime (no binary blobs) by encoding a sine to Ogg/Opus via
/// the same SwiftOGG the decoder uses.
final class VoicePreprocessorTests: XCTestCase {

    /// Write a 16-bit mono PCM WAV (a quiet 440 Hz sine) and return its URL.
    /// Opus only supports 8/12/16/24/48 kHz, so callers must use one of those.
    private func writeWav(frames: Int, sampleRate: Int) throws -> URL {
        var d = Data()
        func u32(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        func u16(_ v: UInt16) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        let dataBytes = frames * 2
        d.append("RIFF".data(using: .ascii)!); d.append(u32(UInt32(36 + dataBytes)))
        d.append("WAVE".data(using: .ascii)!); d.append("fmt ".data(using: .ascii)!)
        d.append(u32(16)); d.append(u16(1)); d.append(u16(1))
        d.append(u32(UInt32(sampleRate))); d.append(u32(UInt32(sampleRate * 2)))
        d.append(u16(2)); d.append(u16(16))
        d.append("data".data(using: .ascii)!); d.append(u32(UInt32(dataBytes)))
        for i in 0..<frames {
            let s = Int16(8000.0 * sin(2.0 * Double.pi * 440.0 * Double(i) / Double(sampleRate)))
            d.append(u16(UInt16(bitPattern: s)))
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("vptest_\(UUID().uuidString).wav")
        try d.write(to: url)
        return url
    }

    /// Synthesise a real Ogg/Opus clip (the shape Telegram delivers).
    private func makeOggOpus(seconds: Double) throws -> Data {
        let wav = try writeWav(frames: Int(16_000 * seconds), sampleRate: 16_000)
        defer { try? FileManager.default.removeItem(at: wav) }
        let oga = FileManager.default.temporaryDirectory
            .appendingPathComponent("vptest_\(UUID().uuidString).oga")
        defer { try? FileManager.default.removeItem(at: oga) }
        try OGGConverter.convertM4aFileToOpusOGG(src: wav, dest: oga)
        return try Data(contentsOf: oga)
    }

    func testDecodesOggOpusVoiceNoteToPCM() throws {
        let ogg = try makeOggOpus(seconds: 0.5)
        guard let decoded = VoicePreprocessor.decode(ogg, oggOpus: true) else {
            return XCTFail("Ogg/Opus voice note failed to decode")
        }
        defer { try? FileManager.default.removeItem(at: decoded.fileURL) }

        XCTAssertEqual(decoded.pcm.count % 4, 0, "PCM must be whole float32 samples")
        let samples = decoded.pcm.count / 4
        // ~0.5 s at 16 kHz ≈ 8000 samples; codec priming/padding makes this loose.
        XCTAssertGreaterThan(samples, 4_000, "decoded clip should carry ~0.5s of audio")
        XCTAssertTrue(FileManager.default.fileExists(atPath: decoded.fileURL.path),
                      "a decodable file must be left on disk for transcription")
    }

    func testDecodesPlainAudioFileWithoutOgg() throws {
        // The non-Opus branch: AVFoundation reads a WAV directly (no SwiftOGG).
        let wav = try writeWav(frames: 8_000, sampleRate: 16_000)
        defer { try? FileManager.default.removeItem(at: wav) }
        let bytes = try Data(contentsOf: wav)

        guard let decoded = VoicePreprocessor.decode(bytes, oggOpus: false, sourceExtension: "wav") else {
            return XCTFail("plain WAV failed to decode via the AVFoundation branch")
        }
        defer { try? FileManager.default.removeItem(at: decoded.fileURL) }
        XCTAssertGreaterThan(decoded.pcm.count / 4, 4_000)
    }

    func testReturnsNilForGarbageOpusBytes() {
        // Non-Ogg bytes on the Opus path must fail cleanly, not crash.
        XCTAssertNil(VoicePreprocessor.decode(Data("definitely not ogg".utf8), oggOpus: true))
    }
}
