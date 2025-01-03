import random
def generic_labels(doc_labels):
    for i in range(len(doc_labels)):
        output_labels = []
        for label in doc_labels[i]:
            if label.startswith("O"):
                output_labels.append(label)
            else:
                output_labels.append(label[2:])
        doc_labels[i] = output_labels
    return doc_labels

def task_splitter(words,labels,label_list):
    random.seed(42)
    tasks = {}
    for labelx in label_list:
        tasks[labelx]=[]
        for word, label in zip(words, labels):
            if labelx in label:
                tasks[labelx].append((words, ["O" if l!=labelx else labelx for l in label]))
        tasks[labelx].shuffle()
    return tasks

            

