import os

results_dir = ['/inputs/test-results-a', '/inputs/test-results-b']

def main():
    for folder in results_dir:
        for filename in os.listdir(folder):
            filepath= os.path.join(folder,filename)
            if os.path.isfile(filepath):
                print("Test results for: " + str(filepath))
                with open(filepath, 'r') as f:
                    print(f.read())
                print('\n\n')

    return

if __name__ == "__main__":
   main()
        
        


