import mdtraj as md
import numpy as np

def clusterByRMSD(trajectory, threshold=None, n_clusters=None, energies=None, centroids=False,
                  max_iter=20):
    """
    Cluster trajectories by RMSD. A target number of clusters can be set and the method will
    find a threshold RMSD distance that generates an approximate number of clusters as possible.
    You can specify a specfic threshold distance to cluster the structures. It can optionally
    take an energy term to order clusters from lowest to higher energy.

    Parameters
    ----------

    trajectory : mdtraj.Trajectory
        Trajectory containing the snapshots to cluster.
    threshold : float
        Cutoff RMSD value used to decide cluster boundaries (nanometers).
    energies : np.ndarray
        Energies of conformations
    centroids : bool
        Return the centroids of each cluster
    max_iter: int
        Number of iterations to find the threshold for the wanted number of clusters.
    Returns
    -------

    clusters : dict
        Dictionary contaning the members of each cluster.
    """

    # Define clusters Dictionary
    clusters = {}

    print('Calculating RMSD matrix')
    n_frames = trajectory.n_frames
    rmsd = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        rmsd[i] = md.rmsd(trajectory, trajectory, frame=i)

    # Summary statistics for RMSD matrix
    mask = np.ones(rmsd.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    average_value = np.average(rmsd[mask])
    max_value = rmsd[mask].max()
    min_value = rmsd[mask].min()
    print()
    print('Average RMSD value: %.4f nm' % average_value)
    print('Maximum RMSD value: %.4f nm' % max_value)
    print('Minimum RMSD value: %.4f nm' % min_value)

    if n_clusters != None:

        if n_clusters > n_frames:
            raise ValueError('Number of clusters larger than the number of indexes.\
            This is just weird!')

        # Define first threshold as the average value of the distance matrix.
        threshold = average_value
        # First guess using the threshold value
        clusters = clusterMatrix(rmsd, threshold)
        # Calculate the step size of the search
        stepsize = (max_value-average_value)/10
        # Try threshold values until the number of clusters is reached
        count = 0
        best_solution = None
        while len(clusters) != n_clusters:
            if len(clusters) < n_clusters:
                threshold = threshold-stepsize
            if len(clusters) > n_clusters:
                threshold = threshold+stepsize
            clusters = clusterMatrix(rmsd, threshold)
            count += 1

            # Store as best solution
            if best_solution == None:
                best_solution = (threshold, np.absolute(len(clusters)-n_clusters))
            # Replace best if a better solution is found.
            else:
                if best_solution[1] > np.absolute(len(clusters)-n_clusters):
                    best_solution = (threshold, np.absolute(len(clusters)-n_clusters))

            if count % 2 == 0:
                stepsize = stepsize/2.0
            if count == max_iter:
                # If best is better than current recalculate with best
                if best_solution[1] < np.absolute(len(clusters)-n_clusters):
                    threshold = best_solution[0]
                    clusters = clusterMatrix(rmsd, threshold)
                break

        print('Selected %s clusters at a threshold of %.6f' % (len(clusters), threshold))

    elif threshold != None:
        # Calculate clusters at the specified threshold
        clusters = clusterMatrix(rmsd, threshold)
    else:
        raise ValueError('You need to specify a threshold or a number of clusters \
        to use this function.')

    # Order clusters by energy
    if not isinstance(energies, type(None)):

        # Store index of each cluster and its energy
        cluster_energies = []
        for c in clusters:
            cluster_energies.append((c, np.min(energies[clusters[c]])))

        # Sort clusters by energy
        sorted_clusters = {}
        for i,c in enumerate(sorted(cluster_energies, key=lambda x:x[1])):
            sorted_clusters[i] = clusters[c[0]]

        clusters = sorted_clusters

    # Return RMSD centroids
    if centroids:
        for i in clusters:
            sub_rmsd = rmsd[clusters[i]][:,clusters[i]]
            centroid_index = np.argmin(np.sum(sub_rmsd, axis=1))
            clusters[i] = clusters[i][centroid_index]

    return clusters

def clusterMatrix(distance_matrix, threshold):
    """
    The function generate clusters from a distance matrix by using a specfied threshold
    value. The process count the member with the largest number of neighbours with
    a distance lower than the threshold value to generate the first cluster. All
    the assigned members are removed from consideration and the process is repeated
    until there is no new models to assign.

    Parameters
    ----------
    distance_matrix : np.ndarray
        Input distance matrix
    threshold : Threshold value to generate clusters

    Returns
    -------
    clusters : dict
        Dictionary containing the cluster members.
    """

    # Define clusters Dictionary
    clusters = {}

    # Mask diagonal indexes from the distance_matrix array
    mask = np.zeros(distance_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, 1)
    m_matrix = np.ma.masked_array(distance_matrix, mask)
    # Generate clusters until no new cluster members are found
    members = distance_matrix.shape[0]
    i = 0
    while np.max(members) > 0:
        # Count cluster with the largest number of members
        members = np.sum(m_matrix <= threshold, axis=1)

        # Check if new clusters are found
        if np.max(members) == 0:
            break

        # Assign indexes of cluster memebers
        clusters[i] = np.ma.where(m_matrix[np.argmax(members)] <= threshold)[0]
        # Add already assigned indexes to mask
        mask[clusters[i],:] = 1
        mask[:,clusters[i]] = 1
        # Update masked array
        m_matrix = np.ma.masked_array(distance_matrix, mask)
        i += 1

    # Add remaining single structure clusters
    for m in set(np.where(mask == False)[0]):
        clusters[i] = np.array([m])
        i += 1

    return clusters
