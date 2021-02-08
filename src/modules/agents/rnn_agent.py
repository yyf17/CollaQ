import torch as th
import torch.nn as nn
import torch.nn.functional as F



class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

    def update_n_agents(self, n_agents):
        pass

if __name__ == '__main__':
    bs = 3
    ts = 2
    na = 2
    os = (3,3)
    input_shape = (bs,ts,na,*os)
    cuda2 = th.device('cuda:2')
    inputs = th.ones(input_shape, dtype=th.float64, device=cuda2)
    args = {
        "rnn_hidden_dim":24
    }
    rnn_agent = RNNAgent(input_shape,args)
    hidden_state = th.ones((args.rnn_hidden_dim,args.rnn_hidden_dim), dtype=th.float64, device=cuda2)
    q,h = rnn_agent.forward(inputs,hidden_state)
    print(q)
    print(h)
#   for project ASN
        # self.move_feat_end = 4
        #
        # self.blood_feat_start = 4 + 5 * self.args.enemies_num + 5 * (self.args.agents_num - 1)
        # # self.blood_feat_start = 5 + 5 * 8 + 5 * 8
        # self.blood_feat_end = self.blood_feat_start + 1
        #
        # self.other_feat_start = 4 + 5 * self.args.enemies_num + 5 * (self.args.agents_num - 1) + 1
        # # self.other_feat_start = 5 + 5 * 8 + 5 * 8 + 1
        #
        # self.enemies_feat_start = 4
        #
        # self.agents_feat_start = 4 + 5 * self.args.enemies_num

        # self_input = th.cat([inputs[:, :self.move_feat_end],
        #                      inputs[:, self.blood_feat_start: self.blood_feat_end],
        #                      inputs[:, self.other_feat_start:]],
        #                     dim=1)
        #
        # enemies_feats = [inputs[:, self.enemies_feat_start + i * 5: self.enemies_feat_start + 5 * (1 + i)]
        #                  for i in range(self.args.enemies_num)]
        #
        # agents_feats = [inputs[:, self.agents_feat_start + i * 5: self.agents_feat_start + 5 * (1 + i)]
        #                 for i in range(self.args.agents_num - 1)]
        #
        # print('normal input: ', inputs)
        # print('-' * 50)
        # print('move feat input: ', inputs[:, :self.move_feat_end])
        # print('-' * 50)
        # print('blood input: ', inputs[:, self.blood_feat_start: self.blood_feat_end])
        # print('-' * 50)
        # print('other feat: ', inputs[:, self.other_feat_start:])
        # print('-' * 50)
        # print('agents feats:', agents_feats)
        # print('-' * 50)
        # print('enemies feats: ',enemies_feats)