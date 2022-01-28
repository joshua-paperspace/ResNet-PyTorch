import os
import json

def main():
    results_dir = '/inputs/test-results'
    accuracy = 0

    for filename in os.listdir(results_dir):
        filepath= os.path.join(results_dir,filename)
        if os.path.isfile(filepath):
            print("Test results for: " + str(filepath))
            with open(filepath, 'r') as f:
                results = json.load(f)
                print(results)
                temp_accuracy = results['accuracy']
                if temp_accuracy > accuracy:
                    model_id = results['model_id']
            print('\n\n')

    with open("/outputs/model-id", "w") as o:
        o.write(model_id)

    return

if __name__ == "__main__":
   main()
        
        


