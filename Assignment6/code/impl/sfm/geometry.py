import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # Normalize coordinates (to points on the normalized image plane)

  # homogeneous coordinates (with additional dimension)
  h_kps1 = np.append(im1.kps, np.ones((im1.kps.shape[0], 1)), 1)
  h_kps2 = np.append(im2.kps, np.ones((im2.kps.shape[0], 1)), 1)

  K_1 = np.linalg.inv(K)

  normalized_kps1 = np.matmul(K_1, h_kps1.T).T / np.matmul(K_1, h_kps1.T).T[:, -1, None]
  normalized_kps2 = np.matmul(K_1, h_kps2.T).T / np.matmul(K_1, h_kps2.T).T[:, -1, None]
  
  # Assemble constraint matrix as equation 2.1
  constraint_matrix = np.zeros((matches.shape[0], 9))
  for i in range(matches.shape[0]):
    # Add the constraints
    constraint_matrix[i] = np.reshape(np.outer(normalized_kps2[matches[i, 1]],\
                                               normalized_kps1[matches[i, 0]]), 9)
  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1, :]

  # Reshape the vectorized matrix to it's proper shape again
  E_hat = np.reshape(vectorized_E_hat, (3, 3))

  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  u, _, vh = np.linalg.svd(E_hat)
  s = np.diag([1, 1, 0])
  E = np.matmul(np.matmul(u, s), vh)

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i, 0], :]
    kp2 = normalized_kps2[matches[i, 1], :]

    #assert(abs(kp1.transpose() @ E @ kp2) < 0.01)
    assert (abs(kp2.transpose() @ E @ kp1) < 0.01)

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations (Determinant should be 1)
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)

  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:, 0]
  im2_corrs = new_matches[:, 1]

  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

  # Filter points behind the first camera
  h_points3D = np.append(points3D, np.ones((points3D.shape[0], 1)), 1)
  cam1 = np.matmul(P1, h_points3D.T).T
  im1_corrs = im1_corrs[cam1[:, -1] > 0]
  im2_corrs = im2_corrs[cam1[:, -1] > 0]
  points3D = points3D[cam1[:, -1] > 0]

  # Filter points behind the second camera
  h_points3D = np.append(points3D, np.ones((points3D.shape[0], 1)), 1)
  cam2 = np.matmul(P2, h_points3D.T).T
  im1_corrs = im1_corrs[cam2[:, -1] > 0]
  im2_corrs = im2_corrs[cam2[:, -1] > 0]
  points3D = points3D[cam2[:, -1] > 0]

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # We use points in the normalized image plane.
  # This removes the 'K' factor from the proj

  h_points2D = np.append(points2D, np.ones((points2D.shape[0], 1)), 1)
  K_1 = np.linalg.inv(K)
  normalized_points2D = (np.matmul(K_1, h_points2D.T).T / np.matmul(K_1, h_points2D.T).T[:, -1, None])[:, :-1]

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:, :3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  image = images[image_name]
  points3D = np.zeros((0, 3))

  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.

  corrs_image = np.zeros(0)
  corrs = {}

  for reg_image_name in registered_images:
    e_matches = GetPairMatches(reg_image_name, image_name, matches)
    new_points3D, corrs_new_reg_image, corrs_new_image = TriangulatePoints(K, images[reg_image_name],\
                                                                           image, e_matches)
    corrs[reg_image_name] = [corrs_new_reg_image, np.arange(points3D.shape[0],\
                                                            points3D.shape[0]+new_points3D.shape[0])]
    points3D = np.append(points3D, new_points3D, axis=0)
    corrs_image = np.append(corrs_image, corrs_new_image, axis=0)

  corrs[image_name] = [corrs_image, np.arange(points3D.shape[0])]

  return points3D, corrs
  
