import mindspore
import mindspore.nn as nn


from Layers import GraphConvolution, pooling_sum_4tensor


class GraphClassifier(nn.Cell):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphClassifier, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.gcn1 = GraphConvolution(in_features=input_dim, out_features=hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GraphConvolution(in_features=hidden_dim, out_features=hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.gcn3 = GraphConvolution(in_features=hidden_dim, out_features=hidden_dim)

        self.fc1 = nn.Dense(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Dense(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Dense(hidden_dim, num_classes)

    def forward(self, adjacency, input_feature, graph_indicator, labels):
        x = self.bn1(input_feature)
        x = self.gcn1(input_feature, adjacency)
        x = self.bn2(x)
        x = self.gcn2(x, adjacency)
        x = self.bn3(x)
        x = self.gcn3(x, adjacency)
        x = pooling_sum_4tensor(x, graph_indicator, labels).cuda()
        x = nn.ReLU(self.fc1(x))
        x = nn.ReLU(self.fc2(x))
        logits = self.fc3(x)
        return logits