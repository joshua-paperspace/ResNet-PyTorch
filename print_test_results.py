import os

results_dir = '/inputs/test-results'

def main():
    for filename in os.listdir(results_dir):
        filepath= os.path.join(results_dir,filename)
        if os.path.isfile(filepath):
            print("Test results for: " + str(filepath))
            with open(filepath, 'r') as f:
                print(f.read())
            print('\n\n')

    return

if __name__ == "__main__":
   main()
        
        


