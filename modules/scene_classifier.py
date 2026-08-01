import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision.models import resnet50, ResNet50_Weights
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class SceneClassifier:
    def __init__(self):
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch/torchvision not available. SceneClassifier disabled.")
            self.model = None
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights).to(self.device)
        self.model.eval()
        self.preprocess = self.weights.transforms()
        self.class_names = self.weights.meta["categories"]
        
        # Simple mapping from ImageNet keywords to scene categories
        self.scene_map = {
            'Urban/city': ['building', 'street', 'cab', 'store', 'skyscraper', 'traffic light'],
            'Suburban/residential': ['picket fence', 'patio', 'home', 'house'],
            'Rural/agricultural': ['tractor', 'plow', 'barn', 'hay', 'farm'],
            'Forest/woodland': ['forest', 'tree', 'log'],
            'Desert/arid': ['desert', 'sand', 'camel'],
            'Coastal/beach': ['beach', 'sandbar', 'promontory', 'pier'],
            'Mountain/alpine': ['alp', 'volcano', 'valley'],
            'Industrial': ['factory', 'crane', 'oil'],
            'Highway/road': ['highway', 'freeway', 'road', 'truck']
        }

    def _map_to_scene(self, top_classes) -> str:
        scores = {k: 0 for k in self.scene_map.keys()}
        for cls_name in top_classes:
            for scene, keywords in self.scene_map.items():
                if any(kw in cls_name.lower() for kw in keywords):
                    scores[scene] += 1
        
        best_match = max(scores.items(), key=lambda x: x[1])
        if best_match[1] > 0:
            return best_match[0]
        return "Unknown"

    def _extract_region_hints(self, top_classes) -> list:
        # Map specific ImageNet classes to likely geographic regions
        region_map = {
            'Asia (East)': ['jinrikisha', 'shoji', 'panda', 'bamboo'],
            'Asia (South/Southeast)': ['stupa', 'indian elephant', 'water buffalo', 'macaque'],
            'Australia/Oceania': ['koala', 'wallaby', 'wombat', 'platypus', 'echidna'],
            'Africa': ['african elephant', 'lion', 'zebra', 'meerkat', 'hippopotamus', 'ostrich', 'dromedary'],
            'South America': ['llama', 'alpaca', 'macaw', 'toucan', 'sloth', 'capybara'],
            'North America': ['bison', 'bald eagle', 'gila monster'],
            'Europe (Italy)': ['gondola'],
            'Polar/Arctic': ['polar bear', 'arctic fox', 'snowmobile', 'dogsled'],
            'Middle East / North Africa': ['mosque', 'dromedary']
        }
        
        hints = set()
        for cls_name in top_classes:
            for region, keywords in region_map.items():
                if any(kw in cls_name.lower() for kw in keywords):
                    hints.add(region)
        return list(hints)

    def classify(self, image_path: str) -> Dict[str, Any]:
        if not TORCH_AVAILABLE or self.model is None:
            return {'status': 'unavailable', 'error': 'Torch not available'}
            
        try:
            img = Image.open(image_path).convert('RGB')
            batch = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.model(batch).squeeze(0).softmax(0)
                
            # Check top 10 for better regional hint extraction
            top_k_prob, top_k_catid = torch.topk(prediction, 10)
            top_classes_list = []
            top_predictions = []
            
            for i in range(top_k_prob.size(0)):
                cls_name = self.class_names[top_k_catid[i]]
                prob = float(top_k_prob[i])
                top_classes_list.append(cls_name)
                if i < 5:  # Keep top 5 for the regular output
                    top_predictions.append({
                        'class': cls_name,
                        'probability': prob
                    })
                
            scene_type = self._map_to_scene(top_classes_list[:5])
            region_hints = self._extract_region_hints(top_classes_list)
            
            return {
                'status': 'success',
                'scene_type': scene_type,
                'confidence': top_predictions[0]['probability'],
                'top_5': top_predictions,
                'region_hints': region_hints
            }
        except Exception as e:
            logger.error(f"Failed to classify scene: {e}")
            return {'status': 'error', 'error': str(e)}
