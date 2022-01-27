import os

results_dir = ['/inputs/test-results-a', '/inputs/test-results-b']

accuracy = 0
# model_path = 'sample_model_path.pth'

ref_dict = {'resnet18-epochs-1.txt': 'dsrsgtp083suuyu'
            ,'resnet34-epochs-1.txt': 'dsrzkbpudtxoz5q'}

model_ref = 'sample-ref'

def main():
    for folder in results_dir:
        for filename in os.listdir(folder):
            filepath= os.path.join(folder,filename)
            if os.path.isfile(filepath):
                print("Test results for: " + str(filepath))
                with open(filepath, 'r') as f:
                    # print(f.read())
                    results = f.read()
                    print(results)
                    temp_accuracy = int(results[-3:-1])
                    if temp_accuracy > accuracy:
                        model_ref = ref_dict[filename]
                print('\n\n')

    with open("/outputs/bestmodel", "w") as o:
        o.write(model_ref)

    return

if __name__ == "__main__":
   main()
        
        


