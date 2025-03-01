import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.mesh_xl_mtp.get_model import MTPMeshXL

def test_mtp_model(model):
    # Set model to eval mode
    model.eval()
    
    # Create a small test batch with mesh data format
    batch_size = 2
    num_vertices = 10
    test_dict = {
        "vertices": torch.randn(batch_size, num_vertices, 3),  # Random 3D vertices
        "faces": torch.randint(0, num_vertices, (batch_size, 8, 3)),  # Random face indices
        # Add other mesh-related fields if required by your tokenizer
    }
    
    print("Testing perplexity calculation...")
    try:
        with torch.no_grad():
            perp_output = model(test_dict, is_eval=True, is_generate=False)
        print("✓ Perplexity calculation successful")
        print(f"Perplexity value: {perp_output['perplexity']:.2f}")
    except Exception as e:
        print("✗ Perplexity calculation failed")
        print(f"Error: {str(e)}")
    
    print("\nTesting generation...")
    try:
        with torch.no_grad():
            gen_output = model(
                test_dict,
                is_eval=True,
                is_generate=True,
                num_return_sequences=2,
                generation_config={
                    'max_length': 15,
                    'do_sample': True,
                    'top_k': 50,
                    'top_p': 0.95,
                    'temperature': 1.0
                }
            )
        print("✓ Generation successful")
        print("Generated sequences shape:", gen_output["input_ids"].shape if "input_ids" in gen_output else "N/A")
    except Exception as e:
        print("✗ Generation failed")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Create args with all necessary parameters
    args = type('Args', (), {
        'codebook_size': 1024,
        'n_discrete_size': 4,  # Added this required parameter
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        # Add any other required args for your MeshTokenizer
    })()

    # Create model
    model = MTPMeshXL(args)
    
    # Run test
    test_mtp_model(model) 