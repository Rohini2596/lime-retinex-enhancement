import cv2
import numpy as np
EPS = 1e-6
def read_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0
def save_image(path, img):
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
def gradient_x(T):
    return np.roll(T, -1, axis=1) - T
def gradient_y(T):
    return np.roll(T, -1, axis=0) - T
def divergence(px, py):
    return (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))
def initial_illumination(img):
    return np.max(img, axis=2)
def compute_weights(T):
    gh = gradient_x(T)
    gv = gradient_y(T)
    Wh = 1 / (np.abs(gh) + EPS)
    Wv = 1 / (np.abs(gv) + EPS)
    return Wh, Wv
def shrink(x, thresh):
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)
def refine_illumination(T_hat, alpha=0.15, max_iter=50):
    T = T_hat.copy()
    Gh = np.zeros_like(T)
    Gv = np.zeros_like(T)
    Zh = np.zeros_like(T)
    Zv = np.zeros_like(T)
    Wh, Wv = compute_weights(T_hat)
    mu = 0.5
    rho = 1.5
    h, w = T.shape
    fy = np.fft.fftfreq(h).reshape(-1,1)
    fx = np.fft.fftfreq(w).reshape(1,-1)
    lap = (2-2*np.cos(2*np.pi*fy)) + (2-2*np.cos(2*np.pi*fx))
    for _ in range(max_iter):
        rhs = 2*T_hat + mu*divergence(Gh - Zh/mu, Gv - Zv/mu)
        T = np.real(np.fft.ifft2(np.fft.fft2(rhs)/(2+mu*lap+EPS)))
        dh = gradient_x(T)
        dv = gradient_y(T)
        Gh = shrink(dh + Zh/mu, alpha*Wh/mu)
        Gv = shrink(dv + Zv/mu, alpha*Wv/mu)
        Zh += mu*(dh-Gh)
        Zv += mu*(dv-Gv)
        mu *= rho
    return np.clip(T, 0, 1)
def enhance(img, T, gamma=0.8):
    T = np.power(T, gamma)
    T = np.repeat(T[:, :, None], 3, axis=2)
    R = img / (T + EPS)
    return np.clip(R, 0, 1)
def denoise(img):
    img = (img*255).astype(np.uint8)
    img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    return img.astype(np.float32)/255
def recomposition(R, Rd, T):
    T = np.repeat(T[:, :, None], 3, axis=2)
    return R*T + Rd*(1-T)