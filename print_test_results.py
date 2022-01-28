import os
import json

results_dir = '/inputs/test-results'

accuracy = 0
# model_path = 'sample_model_path.pth'

# model_id_dict = {'resnet18-epochs-1.txt': 'mosq6dg6bz01aww'
#             ,'resnet34-epochs-1.txt': 'movalaty7v0itu'}

# model_id = 'sample-model-id'



 
# Opening JSON file
# f = open('data.json')
 
# returns JSON object as
# a dictionary
# data = json.load(f)
 
# Iterating through the json
# list
# for i in data['emp_details']:
#     print(i)
 
# Closing file
# f.close()


def main():
    for filename in os.listdir(results_dir):
        filepath= os.path.join(results_dir,filename)
        if os.path.isfile(filepath):
            print("Test results for: " + str(filepath))
            with open(filepath, 'r') as f:
                results = json.load(f)
                # results = f.read()
                print(results)
                temp_accuracy = results['accuracy']
                if temp_accuracy > accuracy:
                    model_id = results[model_id]
            print('\n\n')

    with open("/outputs/model-id", "w") as o:
        o.write(model_id)

    return

if __name__ == "__main__":
   main()
        
        


