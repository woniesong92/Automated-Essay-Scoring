import matplotlib
import matplotlib.pyplot as plt
import pdb
import pickle

# predicted = pickle.load(open("data/set5_svr_predicted_scores.pkl","r"))
# actual = pickle.load(open("data/set5_actual_scores.pkl", "r"))



def draw_distribution(predicted_scores, actual_scores):
    deviation = []
    for i in range(len(predicted_scores)):
        deviation.append(predicted_scores[i] - actual_scores[i])

    num_bins = 40

    # the histogram of the data
    n, bins, patches = plt.hist(deviation, num_bins, normed=1, facecolor='green', alpha=0.5)

    # Tweak spacing to prevent clipping of ylabel
    plt.show()



# draw_distribution(predicted, actual)


