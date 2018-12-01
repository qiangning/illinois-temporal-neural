
# In[1]:

import matplotlib

from myLSTM import lstm_NN_position_embedding

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from random import shuffle

import numpy
import torch.optim as optim

from Baseline_LSTM import *
from ELMo_Cache import *
from TemporalDataSet import *

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return category_i
def evalOnTest(model,testset):
    confusion = numpy.zeros((len(output_labels),len(output_labels)),dtype=int)
    for temprel in testset.temprel_ee:
        embed = model.temprel2embeddingSeq(temprel)
        output = model(embed)
        confusion[output_labels[temprel.label]][categoryFromOutput(output)] += 1
    print(confusion)
    return 1.0*numpy.sum([confusion[i][i] for i in range(4)])/numpy.sum(confusion)
# In[2]:


trainset = temprel_set("data/Output4LSTM_Baseline/trainset.xml")
testset = temprel_set("data/Output4LSTM_Baseline/testset.xml")
options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
# options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 1, dropout=0)
emb_cache = elmo_cache(elmo,"elmo_cache_small.pkl",False)

# In[3]:





# In[4]:

# small:
embedding_dim = 256
lstm_hidden_dim = 64
nn_hidden_dim = 32
position_emb_dim = 16

# medium:
# embedding_dim = 512
# lstm_hidden_dim = 128
# nn_hidden_dim = 64
# position_emb_dim = 32

# original:
# embedding_dim = 1024
# lstm_hidden_dim = 256
# nn_hidden_dim = 128
# position_emb_dim = 64

output_labels = {"BEFORE":0,"AFTER":1,"EQUAL":2,"VAGUE":3}
output_dim = len(output_labels)
batch_size = 1
position2ix = {"B":0,"M":1,"A":2,"E1":3,"E2":4}

# model = lstm_baseline(embedding_dim, lstm_hidden_dim, position_emb_dim, emb_cache, output_dim, batch_size, position2ix)
model = lstm_NN_position_embedding(embedding_dim, lstm_hidden_dim, nn_hidden_dim, position_emb_dim, emb_cache, output_dim, batch_size, position2ix)
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.3)

# In[5]:


torch.manual_seed(1)
criterion = nn.CrossEntropyLoss()
allTempRels = trainset.temprel_ee
all_losses = []
all_accuracies = []

# In[ ]:


start = time.time()
for epoch in range(40):
    print("epoch: %d" % epoch,flush=True)
    current_loss = 0
    shuffle(allTempRels)
    for i in range(trainset.size):
        temprel = allTempRels[i]
        model.zero_grad()
        target = torch.cuda.LongTensor([output_labels[temprel.label]])
        embed = model.temprel2embeddingSeq(temprel)
        output = model(embed)
        loss = criterion(output,target)
        current_loss += loss
        if i % 1000 == 0:
            print("%d/%d: %s %.4f %.4f" %(i,trainset.size, timeSince(start), loss, current_loss),flush=True)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
    all_losses.append(current_loss)
    all_accuracies.append(evalOnTest(model,testset))
    print("Loss at epoch %d: %.4f" %(epoch,current_loss),flush=True)
    print("Acc at epoch %d: %.4f" %(epoch,all_accuracies[-1]),flush=True)

    # plot figures
    plt.figure()
    plt.plot(all_losses)

    plt.tight_layout()
    plt.savefig("figs/elmo/lstm_baseline_all_losses_small3.pdf")
    # plt.show()

    plt.figure()
    plt.plot(all_accuracies)

    plt.tight_layout()
    plt.savefig("figs/elmo/lstm_baseline_all_accuracies_small3.pdf")

    plt.close('all')

confusion = numpy.zeros((len(output_labels),len(output_labels)),dtype=int)

testset = temprel_set("data/Output4LSTM_Baseline/testset.xml")
for temprel in testset.temprel_ee:
    embed = model.temprel2embeddingSeq(temprel)
    output = model(embed)
    confusion[output_labels[temprel.label]][categoryFromOutput(output)] += 1
print(confusion,flush=True)


# In[20]:


print("%d/%d=%.2f\%" %(numpy.sum([confusion[i][i] for i in range(4)]),numpy.sum(confusion),1.0*numpy.sum([confusion[i][i] for i in range(4)])/numpy.sum(confusion)),flush=True)

