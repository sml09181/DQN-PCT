import os
import shutil

model_names = ['ppo', 'a2c', 'dqn3', 'dqn5', 'dqn7']
base_path = "/data/sujin/sujin/GlobalStockAnalyzer/results/451760"

def delete_untrained_folders():
    for model in model_names:
        model_path = os.path.join(base_path, model)
        if not os.path.exists(model_path):
            continue
        
        for folder in os.listdir(model_path):
            folder_path = os.path.join(model_path, folder)
            trained_path = os.path.join(folder_path, "trained")
            model_zip = os.path.join(trained_path, f"{model[:3]}.zip")
            
            if not os.path.exists(model_zip):
                print(f"Deleting folder: {folder_path}, {model_zip}")
                shutil.rmtree(folder_path)

if __name__ == "__main__":
    delete_untrained_folders()
