/**
 * Model source classification and HuggingFace utilities tests.
 *
 * Tests classifyModelInput, HuggingFace URL transforms, mirror fallback,
 * pickBestModelFile, buildHfApiUrl/buildHfFileUrl, and extractModelName
 * with URL inputs.
 */

import { describe, it, expect } from 'vitest';
import { classifyModelInput } from '../src/model-source/classify.js';
import {
  transformHuggingFaceUrl,
  getHuggingFaceMirrorUrl,
  isHuggingFaceUrl,
  buildHfApiUrl,
  buildHfFileUrl,
  pickBestModelFile,
} from '../src/model-source/huggingface.js';
import { extractModelName } from '../src/config/resolver.js';

// ---- classifyModelInput ----

describe('classifyModelInput', () => {
  describe('local paths', () => {
    it('classifies relative path as local-path', () => {
      expect(classifyModelInput('./mobilenet.onnx')).toBe('local-path');
    });

    it('classifies absolute Unix path as local-path', () => {
      expect(classifyModelInput('/home/user/model.onnx')).toBe('local-path');
    });

    it('classifies Windows path as local-path', () => {
      expect(classifyModelInput('C:\\Users\\model\\resnet.onnx')).toBe('local-path');
    });

    it('classifies filename with extension as local-path', () => {
      expect(classifyModelInput('model.onnx')).toBe('local-path');
    });

    it('classifies path with dots in directory as local-path', () => {
      expect(classifyModelInput('./models/v1.2/model.onnx')).toBe('local-path');
    });
  });

  describe('URLs', () => {
    it('classifies https URL as url', () => {
      expect(classifyModelInput('https://huggingface.co/user/repo/resolve/main/model.onnx')).toBe('url');
    });

    it('classifies http URL as url', () => {
      expect(classifyModelInput('http://example.com/model.onnx')).toBe('url');
    });

    it('classifies URL with query params as url', () => {
      expect(classifyModelInput('https://hf.co/user/repo/resolve/main/model.onnx?download=true')).toBe('url');
    });

    it('classifies case-insensitive protocol as url', () => {
      expect(classifyModelInput('HTTPS://example.com/model.onnx')).toBe('url');
    });
  });

  describe('HuggingFace model IDs', () => {
    it('classifies owner/repo as hf-model-id', () => {
      expect(classifyModelInput('user/model')).toBe('hf-model-id');
    });

    it('classifies org/model-name with hyphens as hf-model-id', () => {
      expect(classifyModelInput('microsoft/resnet-50')).toBe('hf-model-id');
    });

    it('classifies with underscores as hf-model-id', () => {
      expect(classifyModelInput('org_name/model_v2')).toBe('hf-model-id');
    });

    it('does NOT classify owner/repo.ext (with dot) as hf-model-id', () => {
      expect(classifyModelInput('user/model.onnx')).toBe('local-path');
    });

    it('does NOT classify multi-level path as hf-model-id', () => {
      expect(classifyModelInput('user/repo/file')).toBe('local-path');
    });
  });

  describe('whitespace handling', () => {
    it('trims leading/trailing whitespace', () => {
      expect(classifyModelInput('  user/model  ')).toBe('hf-model-id');
    });

    it('trims whitespace from URLs', () => {
      expect(classifyModelInput('  https://example.com/model.onnx  ')).toBe('url');
    });
  });
});

// ---- HuggingFace URL utilities ----

describe('transformHuggingFaceUrl', () => {
  it('converts /blob/ to /resolve/', () => {
    const input = 'https://huggingface.co/user/repo/blob/main/model.onnx';
    const expected = 'https://huggingface.co/user/repo/resolve/main/model.onnx';
    expect(transformHuggingFaceUrl(input)).toBe(expected);
  });

  it('leaves /resolve/ URLs unchanged', () => {
    const input = 'https://huggingface.co/user/repo/resolve/main/model.onnx';
    expect(transformHuggingFaceUrl(input)).toBe(input);
  });

  it('leaves non-HF URLs unchanged', () => {
    const input = 'https://example.com/model.onnx';
    expect(transformHuggingFaceUrl(input)).toBe(input);
  });

  it('only replaces the first /blob/ occurrence', () => {
    const input = 'https://huggingface.co/user/repo/blob/main/blob/model.onnx';
    const result = transformHuggingFaceUrl(input);
    expect(result).toBe('https://huggingface.co/user/repo/resolve/main/blob/model.onnx');
  });
});

describe('getHuggingFaceMirrorUrl', () => {
  it('replaces huggingface.co with hf-mirror.com', () => {
    const input = 'https://huggingface.co/user/repo/resolve/main/model.onnx';
    expect(getHuggingFaceMirrorUrl(input)).toBe(
      'https://hf-mirror.com/user/repo/resolve/main/model.onnx',
    );
  });

  it('replaces hf.co with hf-mirror.com', () => {
    const input = 'https://hf.co/user/repo/resolve/main/model.onnx';
    expect(getHuggingFaceMirrorUrl(input)).toBe(
      'https://hf-mirror.com/user/repo/resolve/main/model.onnx',
    );
  });

  it('returns null for non-HF URLs', () => {
    expect(getHuggingFaceMirrorUrl('https://example.com/model.onnx')).toBeNull();
  });
});

describe('isHuggingFaceUrl', () => {
  it('detects huggingface.co', () => {
    expect(isHuggingFaceUrl('https://huggingface.co/user/repo')).toBe(true);
  });

  it('detects hf.co', () => {
    expect(isHuggingFaceUrl('https://hf.co/user/repo')).toBe(true);
  });

  it('detects hf-mirror.com', () => {
    expect(isHuggingFaceUrl('https://hf-mirror.com/user/repo')).toBe(true);
  });

  it('returns false for non-HF URLs', () => {
    expect(isHuggingFaceUrl('https://example.com/model.onnx')).toBe(false);
  });

  it('case insensitive', () => {
    expect(isHuggingFaceUrl('https://HUGGINGFACE.CO/user/repo')).toBe(true);
  });
});

describe('buildHfApiUrl', () => {
  it('builds correct API URL', () => {
    expect(buildHfApiUrl('user/repo')).toBe('https://huggingface.co/api/models/user/repo');
  });
});

describe('buildHfFileUrl', () => {
  it('builds direct download URL for root file', () => {
    expect(buildHfFileUrl('user/repo', 'model.onnx')).toBe(
      'https://huggingface.co/user/repo/resolve/main/model.onnx',
    );
  });

  it('builds direct download URL for nested file', () => {
    expect(buildHfFileUrl('user/repo', 'onnx/model.onnx')).toBe(
      'https://huggingface.co/user/repo/resolve/main/onnx/model.onnx',
    );
  });
});

// ---- pickBestModelFile ----

describe('pickBestModelFile', () => {
  it('picks ONNX file from single-file repo', () => {
    const siblings = [
      { rfilename: 'README.md' },
      { rfilename: 'model.onnx' },
      { rfilename: 'config.json' },
    ];
    expect(pickBestModelFile(siblings)).toBe('model.onnx');
  });

  it('prefers root-level ONNX over nested', () => {
    const siblings = [
      { rfilename: 'onnx/model_quantized.onnx' },
      { rfilename: 'model.onnx' },
    ];
    expect(pickBestModelFile(siblings)).toBe('model.onnx');
  });

  it('prefers shorter name at same depth (non-quantized)', () => {
    const siblings = [
      { rfilename: 'model_int8_quantized.onnx' },
      { rfilename: 'model.onnx' },
    ];
    expect(pickBestModelFile(siblings)).toBe('model.onnx');
  });

  it('picks nested ONNX when no root-level exists', () => {
    const siblings = [
      { rfilename: 'README.md' },
      { rfilename: 'onnx/model.onnx' },
    ];
    expect(pickBestModelFile(siblings)).toBe('onnx/model.onnx');
  });

  it('prefers TFLite when preferTflite is true', () => {
    const siblings = [
      { rfilename: 'model.onnx' },
      { rfilename: 'model.tflite' },
    ];
    expect(pickBestModelFile(siblings, true)).toBe('model.tflite');
  });

  it('falls back to ONNX when preferTflite but no TFLite available', () => {
    const siblings = [
      { rfilename: 'model.onnx' },
      { rfilename: 'README.md' },
    ];
    expect(pickBestModelFile(siblings, true)).toBe('model.onnx');
  });

  it('falls back to TFLite when no ONNX available (default mode)', () => {
    const siblings = [
      { rfilename: 'model.tflite' },
      { rfilename: 'README.md' },
    ];
    expect(pickBestModelFile(siblings)).toBe('model.tflite');
  });

  it('returns null when no model files exist', () => {
    const siblings = [
      { rfilename: 'README.md' },
      { rfilename: 'config.json' },
    ];
    expect(pickBestModelFile(siblings)).toBeNull();
  });

  it('returns null for empty siblings list', () => {
    expect(pickBestModelFile([])).toBeNull();
  });

  it('case insensitive on extension matching', () => {
    const siblings = [
      { rfilename: 'MODEL.ONNX' },
    ];
    expect(pickBestModelFile(siblings)).toBe('MODEL.ONNX');
  });

  it('handles deeply nested files', () => {
    const siblings = [
      { rfilename: 'models/onnx/v2/model.onnx' },
      { rfilename: 'models/onnx/model.onnx' },
    ];
    expect(pickBestModelFile(siblings)).toBe('models/onnx/model.onnx');
  });
});

// ---- extractModelName with URLs ----

describe('extractModelName with URLs', () => {
  it('extracts filename from HuggingFace URL', () => {
    expect(extractModelName('https://huggingface.co/user/repo/resolve/main/model.onnx'))
      .toBe('model');
  });

  it('strips query params before extracting', () => {
    expect(extractModelName('https://hf.co/user/repo/resolve/main/model.onnx?download=true'))
      .toBe('model');
  });

  it('strips hash before extracting', () => {
    expect(extractModelName('https://example.com/models/resnet50.onnx#section'))
      .toBe('resnet50');
  });

  it('extracts from nested path in URL', () => {
    expect(extractModelName('https://huggingface.co/user/repo/resolve/main/onnx/model_q4.onnx'))
      .toBe('model_q4');
  });

  it('handles URL with both query and hash', () => {
    expect(extractModelName('https://example.com/model.onnx?v=2#top'))
      .toBe('model');
  });

  it('handles HuggingFace model ID (owner/repo)', () => {
    expect(extractModelName('user/repo')).toBe('repo');
  });

  it('still handles regular local paths', () => {
    expect(extractModelName('./mobilenet.onnx')).toBe('mobilenet');
    expect(extractModelName('C:\\Users\\model\\resnet.onnx')).toBe('resnet');
  });
});
