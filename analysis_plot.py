import numpy as np
import glob
import pandas as pd
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d as mp3d
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', help='the sub folder under analysis_test_accuracy')
parser.add_argument('-d', '--dimension', help='the dimension of data')
parser.add_argument('-o', '--output', help='output file name')
args = parser.parse_args()

SUB_FOLDER = args.folder
if SUB_FOLDER is None:
    LOG_PATHS = os.path.join('log', 'analysis_test_accuracy', 'landscape', )
else:
    LOG_PATHS = os.path.join('log', 'analysis_test_accuracy', 'landscape',  SUB_FOLDER)

print('log path: {}'.format(LOG_PATHS))

csv_path = os.path.join(LOG_PATHS, '*.csv')
x_dimension = 16 if args.dimension is None else args.dimension
output_file_name = 'ipec_pca_plot' if args.output is None else args.output

def load_data(csv_path, x_dimension):
    files = glob.glob(csv_path)
    data = None
    for file in files:
        loaded_data = pd.read_csv(file, header=None)
        data = loaded_data if data is None else pd.concat([data, loaded_data], ignore_index=True)
    df_x = data.iloc[:, 1:x_dimension + 1]
    df_y = data.iloc[:, x_dimension + 1]
    # get the first 2 pca components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_x.values)
    df_pca_x = pd.DataFrame(pca_result)
    # prepare x, y, z data for 3D plots
    x = df_pca_x.iloc[:, 0].values
    y = df_pca_x.iloc[:, 1].values
    z = df_y.values
    return (x, y, z)

def plot_1d(x, z, output_file_path):
    # 1D plot
    fig = plt.figure(1, figsize=(20, 40))
    # scatter plot
    plt.subplot(211)
    plt.scatter(x, z, c=z, alpha=0.3)
    plt.xlabel('PCA One')
    plt.ylabel('Accuracy')
    plt.title('Scatter Plot')
    # line plot
    plt.subplot(212)
    plt.plot(x, z)
    plt.xlabel('PCA One')
    plt.ylabel('Accuracy')
    plt.title('Line Plot')
    fig.savefig(output_file_path)

def plot_2d(x, y, z, output_file_path):
    # 3D plot
    fig = plt.figure(2, figsize=(20, 40))
    # scatter plot
    ax1 = fig.add_subplot(211, projection='3d')
    ax1.scatter(x, y, z, c=z, marker='o', alpha=0.2)
    ax1.set_xlabel('PCA One')
    ax1.set_ylabel('PCA Two')
    ax1.set_zlabel('Accuracy')
    ax1.set_title('Scatter Plot')
    # line plot
    ax2 = fig.add_subplot(212, projection='3d')
    ax2.plot(x, y, z, alpha=0.2)
    ax2.set_xlabel('PCA One')
    ax2.set_ylabel('PCA Two')
    ax2.set_zlabel('Accuracy')
    ax2.set_title('Line Plot')
    fig.savefig(output_file_path)

def plot_2d_surface(x, y, z, output_file_path):
    # mesh 3D plot
    fig = plt.figure(3, figsize=(20, 20))

    ax = Axes3D(fig)
    m_x, m_y = np.meshgrid(x, y)
    ax.plot_surface(m_x, m_y, z, rstride=1, cstride=1, cmap=cm.viridis)
    ax.set_xlabel('PCA One')
    ax.set_ylabel('PCA Two')
    ax.set_zlabel('Accuracy')
    ax.set_title('Meshgrid surface')
    fig.savefig(output_file_path)

x, y, z = load_data(csv_path, x_dimension)
output_file_path_1d = os.path.join(LOG_PATHS, output_file_name+'_1d.png')
output_file_path_2d = os.path.join(LOG_PATHS, output_file_name+'_2d.png')
output_file_path_2d_surface = os.path.join(LOG_PATHS, output_file_name+'_2d_surface.png')
print('plotting 1d figure')
plot_1d(x, z, output_file_path_1d)
print('plotting 2d figure')
plot_2d(x, y, z, output_file_path_2d)
print('plotting 2d surface figure')
plot_2d_surface(x, y, z, output_file_path_2d_surface)