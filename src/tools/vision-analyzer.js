import sharp from 'sharp';
import { createRequire } from 'module';
const require = createRequire(import.meta.url);
let Tesseract;
try {
  const tesseract = await import('tesseract.js');
  Tesseract = tesseract.Tesseract;
} catch (e) {
  console.error('[Vision] Tesseract not available, OCR disabled');
}

export async function analyzeImage(imagePath) {
  try {
    const metadata = await sharp(imagePath).metadata();
    const stats = await sharp(imagePath).stats();

    const dominantColor = analyzeColors(stats);
    const brightness = stats.channels.reduce((sum, ch) => sum + ch.mean, 0) / (stats.channels.length || 1);

    return {
      dimensions: { width: metadata.width, height: metadata.height },
      format: metadata.format,
      dominantColor,
      brightness,
      isHighRes: (metadata.width && metadata.width >= 1000),
      channels: metadata.channels,
      assessment: generateAssessment(metadata, dominantColor, brightness),
    };
  } catch (e) {
    return { error: 'Failed to analyze image', details: e.message };
  }
}

function analyzeColors(stats) {
  const ch = stats.channels[0] || { mean: 0 };
  const r = stats.channels[0]?.mean || 0;
  const g = stats.channels[1]?.mean || 0;
  const b = stats.channels[2]?.mean || 0;

  if (r > 200 && g < 100 && b < 100) return 'red_dominant';
  if (r < 100 && g > 150 && b < 100) return 'green_dominant';
  if (r > 150 && g > 100 && b < 100) return 'orange_brick_likely';
  if (Math.abs(r - g) < 20 && Math.abs(g - b) < 20) return 'grayscale';
  return 'mixed';
}

function generateAssessment(metadata, dominant, brightness) {
  const assessments = [];
  if (dominant.includes('brick')) assessments.push('image contains warm/brick tones');
  if (brightness < 100) assessments.push('low light conditions possible');
  if (brightness > 200) assessments.push('bright outdoor daylight');
  if ((metadata.width && metadata.width < 500)) assessments.push('low resolution image');
  return assessments.join('; ') || 'neutral assessment';
}

export async function extractText(imagePath) {
  try {
    if (!Tesseract) {
      return { supported: false, error: 'Tesseract not available' };
    }
    const result = await Tesseract.recognize(imagePath, 'eng', {
      logger: (m) => { if (m.status === 'recognizing text') console.error(`[OCR] ${Math.round(m.progress*100)}%`); }
    });
    const text = result.data.text.trim();
    return {
      supported: true,
      text,
      significant: extractSignificantText(text),
      confidence: result.data.confidence,
      words: result.data.words?.length || 0,
    };
  } catch (e) {
    return { supported: false, error: e.message };
  }
}

function extractSignificantText(text) {
  const words = text.split(/\s+/).filter(w => w.length > 3);
  const capitals = words.filter(w => /^[A-Z]/.test(w));
  return capitals.slice(0, 10).join(' ');
}

export async function detectBuildingStyle(imagePath) {
  const analysis = await analyzeImage(imagePath);
  const style = {
    type: 'unknown',
    era: 'unknown',
    materials: [],
    features: []
  };

  if (analysis.dominantColor === 'orange_brick_likely') {
    style.materials.push('brick');
    style.type = 'brick_apartment_or_residential';
    style.era = 'mid_20th_century';
    style.features.push('exposed brick facade');
  }

  if (analysis.dominantColor === 'grayscale') {
    style.materials.push('concrete_or_stucco');
    style.features.push('prefabricated_panels');
  }

  if (analysis.isHighRes) style.features.push('high_detail_visible');

  return style;
}

export async function analyzeVegetation(imagePath) {
  const analysis = await analyzeImage(imagePath);
  const veg = {
    density: 'unknown',
    climate: 'temperate',
    likelyRegion: 'north_america',
    notes: []
  };

  if (analysis.dominantColor === 'green_dominant') {
    veg.density = 'high';
    veg.notes.push('dense foliage visible');
  }

  if (analysis.brightness > 180) {
    veg.climate = 'warm_or_sunny';
    veg.notes.push('bright conditions suggest sunny climate');
  }

  return veg;
}
