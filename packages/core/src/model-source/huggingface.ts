/**
 * HuggingFace URL utilities.
 *
 * Handles:
 * - URL normalization: /blob/ → /resolve/ for direct download
 * - Mirror fallback: hf-mirror.com when primary is slow
 * - Model ID resolution: owner/repo → best available model file URL
 */

/**
 * Transform a HuggingFace URL for direct download.
 *
 * Rewrites /blob/ to /resolve/ which gives the raw file instead of
 * the HTML viewer page. Same pattern as model2webnn.
 *
 * @param url - HuggingFace URL (may use /blob/ path)
 * @returns URL with /blob/ replaced by /resolve/
 */
export function transformHuggingFaceUrl(url: string): string {
  return url.replace(
    /\/(blob)\//,
    '/resolve/',
  );
}

/**
 * Get the HuggingFace mirror URL for a given HuggingFace URL.
 * Used as fallback when primary huggingface.co is slow or unreachable.
 *
 * @param url - Primary HuggingFace URL
 * @returns Mirror URL using hf-mirror.com, or null if not a HF URL
 */
export function getHuggingFaceMirrorUrl(url: string): string | null {
  if (url.includes('huggingface.co')) {
    return url.replace('huggingface.co', 'hf-mirror.com');
  }
  if (url.includes('hf.co')) {
    return url.replace('hf.co', 'hf-mirror.com');
  }
  return null;
}

/**
 * Check if a URL points to HuggingFace.
 */
export function isHuggingFaceUrl(url: string): boolean {
  return /huggingface\.co|hf\.co|hf-mirror\.com/i.test(url);
}

/**
 * Build the HuggingFace API URL to list model files.
 *
 * @param modelId - HuggingFace model ID (e.g., "user/repo")
 * @returns API URL for model info
 */
export function buildHfApiUrl(modelId: string): string {
  return `https://huggingface.co/api/models/${modelId}`;
}

/**
 * Build a direct download URL for a file in a HuggingFace repo.
 *
 * @param modelId - HuggingFace model ID (e.g., "user/repo")
 * @param filename - File path within the repo (e.g., "model.onnx" or "onnx/model.onnx")
 * @returns Direct download URL
 */
export function buildHfFileUrl(modelId: string, filename: string): string {
  return `https://huggingface.co/${modelId}/resolve/main/${filename}`;
}

/** Model file extensions we can work with, in preference order */
const ONNX_EXTENSIONS = ['.onnx'];
const TFLITE_EXTENSIONS = ['.tflite'];

/**
 * Pick the best model file from a list of repo filenames.
 *
 * Preference order:
 * 1. ONNX files (unless engine is litert, then TFLite first)
 * 2. Prefer files in root or "onnx/" directory
 * 3. Prefer non-quantized over quantized (shorter names first)
 *
 * @param siblings - Array of { rfilename: string } from HF API
 * @param preferTflite - If true, prefer TFLite files (for --engine litert)
 * @returns Best matching filename, or null if no model file found
 */
export function pickBestModelFile(
  siblings: Array<{ rfilename: string }>,
  preferTflite = false,
): string | null {
  const files = siblings.map((s) => s.rfilename);

  const primaryExts = preferTflite ? TFLITE_EXTENSIONS : ONNX_EXTENSIONS;
  const fallbackExts = preferTflite ? ONNX_EXTENSIONS : TFLITE_EXTENSIONS;

  // Try primary format first, then fallback
  for (const exts of [primaryExts, fallbackExts]) {
    const candidates = files.filter((f) =>
      exts.some((ext) => f.toLowerCase().endsWith(ext)),
    );

    if (candidates.length === 0) continue;

    // Prefer files in root or "onnx/" directory (shorter paths)
    // Sort by path depth (fewer slashes = closer to root)
    candidates.sort((a, b) => {
      const depthA = (a.match(/\//g) || []).length;
      const depthB = (b.match(/\//g) || []).length;
      if (depthA !== depthB) return depthA - depthB;
      // Same depth: prefer shorter name (less likely to be quantized variant)
      return a.length - b.length;
    });

    return candidates[0];
  }

  return null;
}
