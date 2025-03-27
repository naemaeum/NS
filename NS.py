import numpy as np

def get_node_neighbors(H):
    m, n = H.shape
    CN_to_VNs = [np.where(H[i, :] == 1)[0].tolist() for i in range(m)]
    VN_to_CNs = [np.where(H[:, j] == 1)[0].tolist() for j in range(n)]
    return CN_to_VNs, VN_to_CNs

class NS():
    def __init__(self, H, l_max=25):
        self.H = H
        self.l_max = l_max
        self.num_CN, self.num_VN = H.shape
        self.total_edges = np.sum(H)
        self.CN_to_VNs, self.VN_to_CNs = get_node_neighbors(self.H)

    def c2v(self, m_v2c, c, v):
        indices = self.CN_to_VNs[c]
        arr = np.tanh(0.5 * m_v2c[c, indices])
        idx = indices.index(v)
        prod_ = np.prod(np.delete(arr, idx))
        prod_ = np.clip(prod_, -0.999999, 0.999999)
        return 2 * np.arctanh(prod_)
    
    def v2c(self, L, m_c2v, c, v):
        return L[v] + (np.sum(m_c2v[self.VN_to_CNs[v], v]) - m_c2v[c, v])
    
    def compute_alpha(self, m_c2v, m_v2c, c, v):
        new_m_c2v = self.c2v(m_v2c, c, v)
        return np.abs(new_m_c2v - m_c2v[c, v])
    
    def decode(self, L):
        m_v2c = np.zeros((self.num_CN, self.num_VN))
        for v, c_list in enumerate(self.VN_to_CNs):
            m_v2c[c_list, v] = L[v]
            
        m_c2v = np.zeros((self.num_CN, self.num_VN))
        L_hat = L.copy()
        alpha = np.zeros(self.num_CN)

        for c in range(self.num_CN):
            if self.CN_to_VNs[c]:
                r_list = [self.compute_alpha(m_c2v, m_v2c, c, v) for v in self.CN_to_VNs[c]]
                alpha[c] = max(r_list)

        l = 0
        while l < self.l_max: 
            i = np.argmax(alpha)            
            for v in self.CN_to_VNs[i]:
                m_c2v[i, v] = self.c2v(m_v2c, i, v)
                alpha[i] = 0
                for c in self.VN_to_CNs[v]:
                    if c != i:
                        m_v2c[c, v] = self.v2c(L, m_c2v, c, v)
                        r_val = 0
                        for v_ in self.CN_to_VNs[c]:
                            r_val = max(r_val, self.compute_alpha(m_c2v, m_v2c, c, v_))
                        alpha[c] = r_val            

            for v in range(self.num_VN):
                L_hat[v] = L[v] + np.sum(m_c2v[self.VN_to_CNs[v], v])

            x = np.where(L_hat >= 0, 0, 1)
            if np.all(np.dot(self.H, x) % 2 == 0):
                break            
            l += 1
        return x

H = np.loadtxt('n196(3,6).txt', dtype=int)
m, n = H.shape
coderate = 1 - m / n
decoder = NS(H)
snr_db = 2.0
snr_lin = 10 ** (snr_db / 10.0)
noise_var = 1.0 / (2.0 * snr_lin * coderate)
noise_std = np.sqrt(noise_var)
bit_errors = 0
frame_errors = 0
frame = 0
for i in range(10000):
    tx = np.ones(n)
    noise = np.random.normal(0.0, noise_std, n)
    rx = tx + noise
    llr = (2.0 / noise_var) * rx
    x = decoder.decode(llr)
    bit_error_count = np.sum(x != np.zeros(n))
    if bit_error_count > 0:
        frame_errors += 1
    bit_errors += bit_error_count
    frame += 1
    if i % 10 == 0:
        print(bit_errors, n*frame)
        print(f"Frame {frame}, FER: {frame_errors / frame}, BER: {bit_errors / (n*frame)}")
print(f"Total frames: {frame}, FER: {frame_errors / frame}, BER: {bit_errors / (n*frame)}")
