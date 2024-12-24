import openpyxl
import matplotlib.pyplot as plt
import numpy as np
import itertools

savepath = 'diagrams/'

ExperimentColorMapping = {
    'DefaultTest' : 'black',
    'SkinnyDeep_20E' : 'darkred',
    'SkinnyDeep' : 'red',
    'Wide3Layer_20E' : 'darkolivegreen',
    'Wide3Layer' : 'greenyellow',
    'Wide4Layer_20E' : 'lightseagreen',
    'Wide4Layer' : 'aquamarine',
    'Wide4Layer_Linear_20E' : 'steelblue',
    'Wide4Layer_Linear' : 'dodgerblue'
    }

StatColorMapping = {
    'Training Loss' : 'b',
    'Training Accuracy' : 'g',
    'Test Loss' : 'm',
    'Test Accuracy' : 'y'
    }

class EpochData:
    def __init__(self):
        self.epoch_n = None
        self.training_loss = None
        self.training_accuracy = None
        self.test_loss = None
        self.test_accuracy = None
        
        return
    
    def print_data(self):
        print(f'-------------------------------------')
        print(f'\tEpoch #: {self.epoch_n}')
        print(f'\tTraining Loss: {self.training_loss}')
        print(f'\tTraining Accuracy: {self.training_accuracy}')
        print(f'\tTest Loss: {self.test_loss}')
        print(f'\tTest Accuracy: {self.test_accuracy}')
        print(f'-------------------------------------')
        
        return
    

class ExperimentData:
    def __init__(self, name):
        self.experiment_name = name
        self.epochs = []
        
    def parse_experiment_data(self):
        fpath = f'experiment_data/{self.experiment_name}_data.xlsx'
        experiment_wb = openpyxl.load_workbook(fpath)
        experiment_sheet = experiment_wb.worksheets[0]
        
        for row_cells in experiment_sheet.iter_rows(min_row = 2):
            epoch_data = EpochData()
            epoch_data.epoch_n = row_cells[0].value
            epoch_data.training_loss = row_cells[1].value
            epoch_data.training_accuracy = row_cells[2].value
            epoch_data.test_loss = row_cells[3].value
            epoch_data.test_accuracy = row_cells[4].value
            
            self.epochs.append(epoch_data)
            
        return
    
    def print_data(self):
        print(f'-------------------------------------')
        print(f'-------------------------------------')
        
        print(f'Experiment Name: {self.experiment_name}')
        for epoch in self.epochs:
            epoch.print_data()
            
        print(f'-------------------------------------')
        print(f'-------------------------------------')
            
        return
    
    def get_epochs_arr(self):
        epochs = []
        
        for epoch in self.epochs:
            epochs.append(epoch.epoch_n)
            
        return epochs
    
    def get_training_losses(self):
        training_losses = []
        
        for epoch in self.epochs:
            training_losses.append(epoch.training_loss)
            
        return training_losses
    
    def get_training_accuracies(self):
        training_accuracies = []
        
        for epoch in self.epochs:
            training_accuracies.append(epoch.training_accuracy)
            
        return training_accuracies
    
    def get_test_losses(self):
        test_losses = []
        
        for epoch in self.epochs:
            test_losses.append(epoch.test_loss)
            
        return test_losses
    
    def get_test_accuracies(self):
        test_accuracies = []
        
        for epoch in self.epochs:
            test_accuracies.append(epoch.test_accuracy)
            
        return test_accuracies
    
def TrainingProgressDiagram(experiment_data):
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    
    ax.set_ylim([0, 1])
        

        
    # TODO:
    # Determine a color mapping for each parameter
    
        
    l1 = ax.plot(experiment_data.get_epochs_arr(), experiment_data.get_training_accuracies(), marker = 'o', color = 'r', label = 'Training Accuracy')[0]
    l2 = ax.plot(experiment_data.get_epochs_arr(), experiment_data.get_test_accuracies(), marker = 'o',color = 'g',label = 'Test Accuracy')[0]
    l3 = ax2.plot(experiment_data.get_epochs_arr(), experiment_data.get_training_losses(), marker = 'o', label = 'Training Loss')[0]
    l4 = ax2.plot(experiment_data.get_epochs_arr(), experiment_data.get_test_losses(), marker = 'o', label = 'Test Loss')[0]
        
    lines = [l1, l2, l3, l4]
        
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, bbox_to_anchor = (-.05, 1), loc = 'upper right', fontsize = 'x-small')
        
    ax.grid()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax2.set_ylabel('Loss')
    
    ax2.set_ybound(lower = 0)
        
        
    plt.title(f'Training Summary of {experiment_data.experiment_name}')
        
    #plt.show()
    plt.savefig(f'TrainingSummary_{experiment_data.experiment_name}.png')
    plt.close()
        
    return
        
def TrainingAccuracyComparison_7E(experiments):
    for experiment in experiments:
        assert(len(experiment.get_epochs_arr()) == 7)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lines = []
    for experiment in experiments:
        line = ax.plot(experiment.get_epochs_arr(), experiment.get_training_accuracies(), marker = 'o', color = ExperimentColorMapping[experiment.experiment_name], label = experiment.experiment_name)[0]
        
        lines.append(line)
        
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, bbox_to_anchor = (-.05, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Accuracy')
    plt.title(f'Training Accuracies for 7 Epoch Models')
    
    #plt.show()
    plt.savefig(f'{savepath}/TrainingAccuracyComparison_7E/TrainingAccuracyComparison_7E.png')
    plt.close()
    
    return

def TrainingLossComparison_7E(experiments):
    for experiment in experiments:
        assert(len(experiment.get_epochs_arr()) == 7)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lines = []
    for experiment in experiments:
        line = ax.plot(experiment.get_epochs_arr(), experiment.get_training_losses(), marker = 'o', color = ExperimentColorMapping[experiment.experiment_name], label = experiment.experiment_name)[0]
        
        lines.append(line)
        
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, bbox_to_anchor = (-.05, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_ybound(lower = 0)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss')
    plt.title(f'Training Losses for 7 Epoch Models')
    
    #plt.show()
    plt.savefig(f'{savepath}/TrainingLossComparison_7E/TrainingLossComparison_7E.png')
    plt.close()
    
    return

def TestAccuracyComparison_7E(experiments):
    for experiment in experiments:
        assert(len(experiment.get_epochs_arr()) == 7)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lines = []
    for experiment in experiments:
        line = ax.plot(experiment.get_epochs_arr(), experiment.get_test_accuracies(), marker = 'o', color = ExperimentColorMapping[experiment.experiment_name], label = experiment.experiment_name)[0]
        
        lines.append(line)
        
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, bbox_to_anchor = (-.05, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Testing Accuracy')
    plt.title(f'Testing Accuracies for 7 Epoch Models')
    
    #plt.show()
    plt.savefig(f'{savepath}/TestAccuracyComparison_7E/TestAccuracyComparison_7E.png')
    plt.close()
    
    return

def TestLossComparison_7E(experiments):
    for experiment in experiments:
        assert(len(experiment.get_epochs_arr()) == 7)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lines = []
    for experiment in experiments:
        line = ax.plot(experiment.get_epochs_arr(), experiment.get_test_losses(), marker = 'o', color = ExperimentColorMapping[experiment.experiment_name], label = experiment.experiment_name)[0]
        
        lines.append(line)
        
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, bbox_to_anchor = (-.05, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_ybound(lower = 0)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Testing Loss')
    plt.title(f'Testing Losses for 7 Epoch Models')
    
    #plt.show()
    plt.savefig(f'{savepath}/TestLossComparison_7E/TestLossComparison_7E.png')
    plt.close()
    
    return
    
def TrainingAccuracyComparison_20E(experiments):
    for experiment in experiments:
        assert(len(experiment.get_epochs_arr()) == 20)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lines = []
    for experiment in experiments:
        line = ax.plot(experiment.get_epochs_arr(), experiment.get_training_accuracies(), marker = 'o', color = ExperimentColorMapping[experiment.experiment_name], label = experiment.experiment_name)[0]
        
        lines.append(line)
        
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, bbox_to_anchor = (-.05, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Accuracy')
    plt.title(f'Training Accuracies for 20 Epoch Models')
    
    #plt.show()
    plt.savefig(f'{savepath}/TrainingAccuracyComparison_20E/TrainingAccuracyComparison_20E.png')
    plt.close()
    
    return

def TrainingLossComparison_20E(experiments):
    for experiment in experiments:
        assert(len(experiment.get_epochs_arr()) == 20)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lines = []
    for experiment in experiments:
        line = ax.plot(experiment.get_epochs_arr(), experiment.get_training_losses(), marker = 'o', color = ExperimentColorMapping[experiment.experiment_name], label = experiment.experiment_name)[0]
        
        lines.append(line)
        
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, bbox_to_anchor = (-.05, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_ybound(lower = 0)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss')
    plt.title(f'Training Losses for 20 Epoch Models')
    
    #plt.show()
    plt.savefig(f'{savepath}/TrainingLossComparison_20E/TrainingLossComparison_20E.png')
    plt.close()
    
    return

def TestAccuracyComparison_20E(experiments):
    for experiment in experiments:
        assert(len(experiment.get_epochs_arr()) == 20)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lines = []
    for experiment in experiments:
        line = ax.plot(experiment.get_epochs_arr(), experiment.get_test_accuracies(), marker = 'o', color = ExperimentColorMapping[experiment.experiment_name], label = experiment.experiment_name)[0]
        
        lines.append(line)
        
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, bbox_to_anchor = (-.05, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Testing Accuracy')
    plt.title(f'Testing Accuracies for 20 Epoch Models')
    
    #plt.show()
    plt.savefig(f'{savepath}/TestAccuracyComparison_20E/TestAccuracyComparison_20E.png')
    plt.close()
    
    return

def TestLossComparison_20E(experiments):
    for experiment in experiments:
        assert(len(experiment.get_epochs_arr()) == 20)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    lines = []
    for experiment in experiments:
        line = ax.plot(experiment.get_epochs_arr(), experiment.get_test_losses(), marker = 'o', color = ExperimentColorMapping[experiment.experiment_name], label = experiment.experiment_name)[0]
        
        lines.append(line)
        
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, bbox_to_anchor = (-.05, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_ybound(lower = 0)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Testing Loss')
    plt.title(f'Testing Losses for 20 Epoch Models')
    
    #plt.show()
    plt.savefig(f'{savepath}/TestLossComparison_20E/TestLossComparison_20E.png')
    plt.close()

    return

def CompareStats(experiment1, experiment2):
    assert(len(experiment1.get_epochs_arr()) == len(experiment2.get_epochs_arr()))
    
    CompareTrainingLoss(experiment1, experiment2)
    CompareTrainingAccuracy(experiment1, experiment2)
    CompareTestLoss(experiment1, experiment2)
    CompareTestAccuracy(experiment1, experiment2)
    BarComparison(experiment1, experiment2)
    
    return

def CompareTrainingLoss(experiment1, experiment2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    exp1_training_loss_curve = ax.plot(experiment1.get_epochs_arr(), experiment1.get_training_losses(), marker = 'o', color = ExperimentColorMapping[experiment1.experiment_name], label = experiment1.experiment_name)[0]
    exp2_training_loss_curve = ax.plot(experiment2.get_epochs_arr(), experiment2.get_training_losses(), marker = 'o', color = ExperimentColorMapping[experiment2.experiment_name], label = experiment2.experiment_name)[0]
    
    curves = [exp1_training_loss_curve, exp2_training_loss_curve]
    labels = [c.get_label() for c in curves]
    ax.legend(curves, labels, bbox_to_anchor = (-0.005, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_ybound(lower = 0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    plt.title(f'Training Losses for {experiment1.experiment_name} and {experiment2.experiment_name}')
    
    #plt.show()
    plt.savefig(f'{savepath}/CompareStats/CompareTrainingLoss_{experiment1.experiment_name}_{experiment2.experiment_name}.png')
    plt.close()
    
    return

def CompareTrainingAccuracy(experiment1, experiment2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    exp1_training_loss_curve = ax.plot(experiment1.get_epochs_arr(), experiment1.get_training_accuracies(), marker = 'o', color = ExperimentColorMapping[experiment1.experiment_name], label = experiment1.experiment_name)[0]
    exp2_training_loss_curve = ax.plot(experiment2.get_epochs_arr(), experiment2.get_training_accuracies(), marker = 'o', color = ExperimentColorMapping[experiment2.experiment_name], label = experiment2.experiment_name)[0]
    
    curves = [exp1_training_loss_curve, exp2_training_loss_curve]
    labels = [c.get_label() for c in curves]
    ax.legend(curves, labels, bbox_to_anchor = (-0.005, 1), loc = 'upper right', fontsize = 'x-small')
    
    
    ax.grid()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy')
    plt.title(f'Training Accuracies for {experiment1.experiment_name} and {experiment2.experiment_name}')
    
    #plt.show()
    plt.savefig(f'{savepath}/CompareStats/CompareTrainingAccuracy_{experiment1.experiment_name}_{experiment2.experiment_name}.png')
    plt.close()

    return

def CompareTestLoss(experiment1, experiment2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    exp1_training_loss_curve = ax.plot(experiment1.get_epochs_arr(), experiment1.get_test_losses(), marker = 'o', color = ExperimentColorMapping[experiment1.experiment_name], label = experiment1.experiment_name)[0]
    exp2_training_loss_curve = ax.plot(experiment2.get_epochs_arr(), experiment2.get_test_losses(), marker = 'o', color = ExperimentColorMapping[experiment2.experiment_name], label = experiment2.experiment_name)[0]
    
    curves = [exp1_training_loss_curve, exp2_training_loss_curve]
    labels = [c.get_label() for c in curves]
    ax.legend(curves, labels, bbox_to_anchor = (-0.005, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_ybound(lower = 0)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    plt.title(f'Test Losses for {experiment1.experiment_name} and {experiment2.experiment_name}')
    
    #plt.show()
    plt.savefig(f'{savepath}/CompareStats/CompareTestLoss_{experiment1.experiment_name}_{experiment2.experiment_name}.png')
    plt.close()
    
    return

def CompareTestAccuracy(experiment1, experiment2):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    exp1_training_loss_curve = ax.plot(experiment1.get_epochs_arr(), experiment1.get_test_accuracies(), marker = 'o', color = ExperimentColorMapping[experiment1.experiment_name], label = experiment1.experiment_name)[0]
    exp2_training_loss_curve = ax.plot(experiment2.get_epochs_arr(), experiment2.get_test_accuracies(), marker = 'o', color = ExperimentColorMapping[experiment2.experiment_name], label = experiment2.experiment_name)[0]
    
    curves = [exp1_training_loss_curve, exp2_training_loss_curve]
    labels = [c.get_label() for c in curves]
    ax.legend(curves, labels, bbox_to_anchor = (-0.005, 1), loc = 'upper right', fontsize = 'x-small')
    
    ax.grid()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Testing Accuracy')
    plt.title(f'Testing Accuracies for {experiment1.experiment_name} and {experiment2.experiment_name}')
    
    #plt.show()
    plt.savefig(f'{savepath}/CompareStats/CompareTestAccuracy_{experiment1.experiment_name}_{experiment2.experiment_name}.png')
    plt.close()
    
    return

def BarSummary(experiment):
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    
    (loss_stat, accuracy_stat) = GetTerminalStats(experiment)
    
    loss_labels = ['Training Loss', 'Test Loss']
    accuracy_labels = ['Training Accuracy', 'Test Accuracy']
    
    
    loss_colors  = [StatColorMapping[x] for x in loss_labels]
    accuracy_colors = [StatColorMapping[x] for x in accuracy_labels]
    
    ax.bar(loss_labels, loss_stat, color = loss_colors)
    ax2.bar(accuracy_labels, accuracy_stat, color = accuracy_colors)
    
    ax.set_ylabel('Loss')
    ax.set_ybound(lower = 0)
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    
    plt.title(f'Training Statistic Bar Plot Summary of {experiment.experiment_name}')
    
    
    #plt.show()
    plt.savefig(f'{savepath}/BarSummary/{experiment.experiment_name}.png')
    plt.close()
    
    return

def BarComparison(experiment1, experiment2):
    assert(len(experiment1.epochs) == len(experiment2.epochs))

    e1_losses, e1_acc = GetTerminalStats(experiment1)
    e2_losses, e2_acc = GetTerminalStats(experiment2)
    
    width = 0.2
    
    x_axis1 = np.array([0, 1])
    x_axis2 = np.array([2, 3])
    
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    
    e1_colors = [ExperimentColorMapping[experiment1.experiment_name]] * 2
    e2_colors = [ExperimentColorMapping[experiment2.experiment_name]] * 2
    
    c1 = ax.bar(x_axis1 - width, e1_losses, 0.4, label = ['Train Loss', 'Test Loss'], color = e1_colors)
    c2 = ax.bar(x_axis1 + width, e2_losses, 0.4, label = ['Train Loss', 'Test Loss'], color = e2_colors)
    
    c3 = ax2.bar(x_axis2 - width, e1_acc, 0.4, label = ['Training Accuracy', 'Test Accuracy'], color = e1_colors)
    c4 = ax2.bar(x_axis2 + width, e2_acc, 0.4, label = ['Training Accuracy', 'Test Accuracy'], color = e2_colors)

    loss_curves = [c1, c2]  
    #acc_curves = [c3, c4]
    experiment_labels = [f'{experiment1.experiment_name}', 
                         f'{experiment2.experiment_name}']
    
    plt.xticks(np.arange(0, 4), ['Training Loss', 'Test Loss', 'Training Accuracy', 'Test Accuracy'])
    
    # No need to label accuracy curves because we are coloring using experiment name.
    ax.legend(loss_curves, experiment_labels, bbox_to_anchor = (0.05, 1.1), loc = 'upper right', fontsize = 'x-small')
    ax.set_ylabel('Loss')
    ax.set_ybound(lower = 0)
    
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    
    plt.title(f'Final Epoch Statistics for {experiment1.experiment_name} and {experiment2.experiment_name}')

    #plt.show()
    plt.savefig(f'{savepath}/CompareStats/BarComparison_{experiment1.experiment_name}_{experiment2.experiment_name}.png')
    plt.close()
    
    # TODO:
    # Bisecting line between loss and accuracy plots?

    return

    

def GetTerminalStats(experiment):
    train_loss = experiment.get_training_losses()
    train_acc = experiment.get_training_accuracies()
    test_loss = experiment.get_test_losses()
    test_acc = experiment.get_test_accuracies()
    
    loss_stat = [train_loss[-1], test_loss[-1]]
    accuracy_stat = [train_acc[-1], test_acc[-1]]
    
    return (loss_stat, accuracy_stat)
    

def load_data():
    e7_experiment_names = [
        'DefaultTest',
        'SkinnyDeep',
        'Wide3Layer',
        'Wide4Layer',
        'Wide4Layer_Linear'
        ]
    
    e20_experiment_names = [
        'SkinnyDeep_20E',
        'Wide3Layer_20E',
        'Wide4Layer_20E',
        'Wide4Layer_Linear_20E'
        ]
    
    e7_experiments = []
    
    for name in e7_experiment_names:
        experiment = ExperimentData(name)
        experiment.parse_experiment_data()
        e7_experiments.append(experiment)

        
    TrainingAccuracyComparison_7E(e7_experiments)
    TrainingLossComparison_7E(e7_experiments)
    TestAccuracyComparison_7E(e7_experiments)
    TestLossComparison_7E(e7_experiments)
    
    e20_experiments = []
    
    for name in e20_experiment_names:
        experiment = ExperimentData(name)
        experiment.parse_experiment_data()
        e20_experiments.append(experiment)
        
    return e7_experiments, e20_experiments

    


if __name__ == '__main__':
    '''
    dt_data = ExperimentData('DefaultTest')
    dt_data.parse_experiment_data()
    
    w4ll_data = ExperimentData('Wide4Layer_Linear')
    w4ll_data.parse_experiment_data()
    
    #BarSummary(dt_data)
    BarComparison(dt_data, w4ll_data)
    '''
    
    e7_exps, e20_exps = load_data()
    
    e7_combinations = list(itertools.combinations(e7_exps, 2))
    e20_combinations = list(itertools.combinations(e20_exps, 2))
    
    

    # Comparison in-epoch-size plots
    TrainingLossComparison_7E(e7_exps)
    TrainingAccuracyComparison_7E(e7_exps)
    TestLossComparison_7E(e7_exps)
    TestAccuracyComparison_7E(e7_exps)
    
    TrainingLossComparison_20E(e20_exps)
    TrainingAccuracyComparison_20E(e20_exps)
    TestLossComparison_20E(e20_exps)
    TestAccuracyComparison_20E(e20_exps)
    
    # Direct comparison plots
    for (exp1, exp2) in e7_combinations:
        CompareStats(exp1, exp2)
        
    for (exp1, exp2) in e20_combinations:
        CompareStats(exp1, exp2)
        
    # BarSummary plots for each experiment
    for exp in e7_exps:
        BarSummary(exp)
        
    for exp in e20_exps:
        BarSummary(exp)
    

    '''
    TrainingAccuracyComparison_20E(e20_experiments)
    TrainingLossComparison_20E(e20_experiments)
    TestAccuracyComparison_20E(e20_experiments)
    TestLossComparison_20E(e20_experiments)
    '''

    '''
    experiment_name = 'DefaultTest'
    dt_data = ExperimentData(experiment_name)
    dt_data.parse_experiment_data()
    #print(dt_data.get_training_losses())
    #dt_data.print_data()
    #TrainingProgressDiagram(dt_data)
    
    w4ll = 'Wide4Layer_Linear'
    w4ll_data = ExperimentData(w4ll)
    w4ll_data.parse_experiment_data()
    #TrainingProgressDiagram(w4ll_data)
    
    TrainingAccuracyComparison_7E([dt_data, w4ll_data])
    '''