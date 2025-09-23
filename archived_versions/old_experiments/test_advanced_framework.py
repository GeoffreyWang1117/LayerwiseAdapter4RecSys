"""
Simplified Advanced Framework Test

This script tests each advanced module independently to validate functionality.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import traceback
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_adaptive_layer_selector():
    """Test the adaptive layer selector module"""
    logger.info("Testing Adaptive Layer Selector...")
    
    try:
        from core.adaptive_layer_selector import AdaptiveLayerSelector, TaskConfig
        
        # Create selector
        selector = AdaptiveLayerSelector()
        
        # Test task config
        task_config = TaskConfig(
            dataset_name='test_dataset',
            task_type='collaborative_filtering',  
            num_users=1000,
            num_items=500,
            embedding_dim=64,
            target_layers=[2, 4, 6, 8]
        )
        
        # Test importance prediction
        importance_profile = selector.predict_layer_importance(task_config)
        
        logger.info(f"✅ Adaptive Layer Selector: Predicted {len(importance_profile.layer_importances)} layer importances")
        return True
        
    except Exception as e:
        logger.error(f"❌ Adaptive Layer Selector failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_multi_dataset_validator():
    """Test the multi-dataset validator module"""
    logger.info("Testing Multi-Dataset Validator...")
    
    try:
        from core.multi_dataset_validator import MultiDatasetValidator
        
        # Create validator
        validator = MultiDatasetValidator()
        
        # Create sample datasets
        datasets = {}
        for name in ['amazon_sample', 'movielens_sample']:
            n_users, n_items = 100, 50
            n_interactions = 200
            
            users = np.random.randint(0, n_users, n_interactions)
            items = np.random.randint(0, n_items, n_interactions)  
            ratings = np.random.uniform(1, 5, n_interactions)
            
            datasets[name] = pd.DataFrame({
                'user_id': users,
                'item_id': items,
                'rating': ratings
            })
        
        # Test validation (with minimal data)
        results = validator.validate_cross_dataset(datasets, use_sample_data=True)
        
        logger.info(f"✅ Multi-Dataset Validator: Validated {len(datasets)} datasets")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-Dataset Validator failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_visualization_suite():
    """Test the visualization suite module"""
    logger.info("Testing Visualization Suite...")
    
    try:
        from core.visualization_suite_fixed import LayerwiseVisualizationSuite
        
        # Create visualization suite
        viz_suite = LayerwiseVisualizationSuite()
        
        # Test visualization creation
        layer_importances = np.random.rand(8)
        layer_names = [f"Layer_{i}" for i in range(8)]
        
        # Create test directory
        test_dir = Path("test_results")
        test_dir.mkdir(exist_ok=True)
        
        # Test importance distribution plot
        viz_suite.plot_layer_importance_distribution(
            layer_importances=layer_importances,
            layer_names=layer_names,
            save_path=test_dir / "test_importance.png"
        )
        
        logger.info("✅ Visualization Suite: Created test visualizations")
        return True
        
    except Exception as e:
        logger.error(f"❌ Visualization Suite failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_rl_layer_optimizer():
    """Test the RL layer optimizer module"""
    logger.info("Testing RL Layer Optimizer...")
    
    try:
        from core.rl_layer_optimizer import RLLayerOptimizer
        
        # Create RL optimizer
        rl_optimizer = RLLayerOptimizer()
        
        # Test with minimal training
        num_users, num_items = 100, 50
        
        # Quick optimization test (1 episode)
        history = rl_optimizer.optimize_layer_configuration(
            num_users=num_users,
            num_items=num_items,
            num_episodes=1
        )
        
        # Test policy
        optimal_layers = rl_optimizer.get_optimal_layer_selection(
            num_users=num_users,
            num_items=num_items
        )
        
        logger.info(f"✅ RL Layer Optimizer: Optimized layers {optimal_layers}")
        return True
        
    except Exception as e:
        logger.error(f"❌ RL Layer Optimizer failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_multi_task_adapter():
    """Test the multi-task adapter module"""
    logger.info("Testing Multi-Task Adapter...")
    
    try:
        from core.multi_task_adapter import MultiTaskLayerwiseAdapter
        from core.adaptive_layer_selector import TaskConfig
        
        # Create multi-task adapter
        adapter = MultiTaskLayerwiseAdapter()
        
        # Create sample task configs
        task_configs = []
        for i, dataset_name in enumerate(['task1', 'task2']):
            config = TaskConfig(
                dataset_name=dataset_name,
                task_type='collaborative_filtering',
                num_users=100 + i*50,
                num_items=50 + i*25,
                embedding_dim=64
            )
            task_configs.append(config)
        
        # Test multi-task training (minimal)
        results = adapter.train_multi_task(
            task_configs=task_configs,
            num_epochs=1
        )
        
        logger.info("✅ Multi-Task Adapter: Completed multi-task training")
        return True
        
    except Exception as e:
        logger.error(f"❌ Multi-Task Adapter failed: {e}")
        logger.error(traceback.format_exc())
        return False


def test_sota_comparison():
    """Test the SOTA comparison module"""
    logger.info("Testing SOTA Comparison...")
    
    try:
        from core.sota_comparison import SOTAComparisonFramework
        
        # Create SOTA comparison framework
        sota = SOTAComparisonFramework()
        
        # Create minimal sample data
        n_users, n_items = 50, 25
        n_interactions = 100
        
        users = np.random.randint(0, n_users, n_interactions)
        items = np.random.randint(0, n_items, n_interactions)
        ratings = np.random.uniform(1, 5, n_interactions)
        
        df = pd.DataFrame({
            'user_id': users,
            'item_id': items,
            'rating': ratings
        })
        
        # Test one model comparison (quick test)
        sota.model_configs = {
            'NCF': sota.model_configs['NCF']  # Test just one model
        }
        sota.model_classes = {
            'NCF': sota.model_classes['NCF']
        }
        
        results = sota.run_comparison(df, 'test_dataset')
        
        logger.info("✅ SOTA Comparison: Completed model comparison")
        return True
        
    except Exception as e:
        logger.error(f"❌ SOTA Comparison failed: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """Run all module tests"""
    
    print("🧪 Advanced Layerwise Adapter - Module Testing")
    print("=" * 50)
    
    # Test modules
    tests = [
        ("Adaptive Layer Selector", test_adaptive_layer_selector),
        ("Multi-Dataset Validator", test_multi_dataset_validator), 
        ("Visualization Suite", test_visualization_suite),
        ("RL Layer Optimizer", test_rl_layer_optimizer),
        ("Multi-Task Adapter", test_multi_task_adapter),
        ("SOTA Comparison", test_sota_comparison)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔬 Testing {test_name}...")
        try:
            success = test_func()
            results[test_name] = success
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"   {status}")
            
        except Exception as e:
            results[test_name] = False
            print(f"   ❌ FAILED: {e}")
    
    # Summary
    print(f"\n📊 Test Summary:")
    passed = sum(results.values())
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Advanced framework is ready.")
    else:
        print("⚠️  Some tests failed. Check logs for details.")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"module_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = main()
