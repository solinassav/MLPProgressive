import os
def summary(model, acc, cost):
    nInput = model.nInputDim
    nOutput = model.nOutputDim
    nHiddenLayer = model.nHidden
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    layerSpace = max(nHiddenLayer) / 3
    nLayer = nHiddenLayer.__len__()
    file = open(model.networkName + "Summary.tex", "w")
    file.write("\documentclass[article]{report}\n" +
               "\\usepackage{tikz}\n" +
               "\\begin{document}\n" +
               "\\author{MLP Solinas}\n" +
               "\\title{" + model.networkName + "\nStructure and data}\n" +
               "\maketitle\n" +
               "\pagestyle{empty}\n" +
               "\def\layersep{" + str(layerSpace) + "}\n" +
               "\\begin{tikzpicture}[shorten >=1pt,->,draw=black!50, node distance=\layersep]\n" +
               "\t\\tikzstyle{every pin edge}=[<-,shorten <=1pt]\n" +
               "\t\\tikzstyle{neuron}=[circle,fill=black!25,minimum size=17pt,inner sep=0pt]\n" +
               "\t\\tikzstyle{input neuron}=[neuron, fill=green!50];\n" +
               "\t\\tikzstyle{output neuron}=[neuron, fill=red!50];\n" +
               "\t\\tikzstyle{hidden neuron}=[neuron, fill=blue!50];\n" +
               "\t\\tikzstyle{annot} = [text width=4em, text centered]\n")
    file.write("\t\\foreach \\name / \y in {1,...," + str(nInput) + "}\n" +
               "\t\t\\node[input neuron,pin = {[pin edge={<-}]left:}] (I-\\name) at (0,-\y) {};\n")
    for i in range(0, nLayer):
        if (i == 0):
            diff = nHiddenLayer.__getitem__(i) - nInput
        else:
            diff += nHiddenLayer.__getitem__(i) - nHiddenLayer.__getitem__(i - 1)
        file.write("\t\\foreach \\name / \y in {1,...," + str(nHiddenLayer.__getitem__(i)) + "}\n" +
                   "\t\t \path[yshift=" + str(diff / 2) + "cm]\n" +
                   "\t\t\tnode[hidden neuron] (" + alphabet.__getitem__(i) + "-\\name) at (" + str(
            i + 1) + "*" + "\layersep  ,-\y cm) {};\n")
    diff += nOutput - nHiddenLayer.__getitem__(nLayer - 1)
    file.write("\t\\foreach \\name / \y in {1,...," + str(nOutput) + "}\n" +
               "\t\t \path[yshift=" + str(diff / 2) + "cm]\n" +
               "\t\t\tnode[output neuron] (O-\\name) at (" + str(
        nLayer + 1) + "*" + "\layersep  ,-\y cm - \layersep) {};\n")
    file.write("\t\\foreach \source in {1,...," + str(nInput) + "}\n" +
               "\t\t\\foreach \dest in {1,...," + str(nHiddenLayer.__getitem__(0)) + "}\n" +
               "\t\t\t\path (I-\source) edge (" + str(alphabet.__getitem__(0)) + "-\dest );\n")
    for i in range(1, nLayer):
        file.write("\t\\foreach \source in {1,...," + str(nHiddenLayer.__getitem__(i - 1)) + "}\n" +
                   "\t\t\\foreach \dest in {1,...," + str(nHiddenLayer.__getitem__(i)) + "}\n" +
                   "\t\t\t\path (" + str(alphabet.__getitem__(i - 1)) + "-\source) edge (" + alphabet.__getitem__(
            i) + "-\dest );\n")
    file.write("\\foreach \source in {1,...," + str(nHiddenLayer.__getitem__(nLayer - 1)) + "}\n" +
               "\t\\foreach \dest in {1,...," + str(nOutput) + "}\n" +
               "\t\t\path (" + str(alphabet.__getitem__(nLayer - 1)) + "-\source) edge (O-\dest );\n")
    file.write("\end{tikzpicture}\n" +
               "\end{document}\n")
    file.close()
    os.system("pdflatex " + model.networkName + "Summary.tex")