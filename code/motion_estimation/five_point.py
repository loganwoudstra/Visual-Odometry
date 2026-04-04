from motion_estimation import EssentialMatrixEstimator
import numpy as np
from itertools import product
import sympy as sp

class FivePointEstimator(EssentialMatrixEstimator):
    def __init__(self, K):
        super().__init__(K)
        
    def solve_cubic_contraints(self, pts1, pts2, nullspace, grid_search_res=10):
        """
        given basis vectors of null space F0, F1, F2, F3
        we have that F = aF0 + bF1 + cF2 + F3.
        Use grid search to find best F (minimzes epipoalr/sampson constraint)
        """
        F0, F1, F2, F3 = [nullspace[:, i].reshape(3, 3) for i in range(4)]
        
        coeff_range = np.linspace(-1, 1, grid_search_res)
        # num_candidates = len(coeff_range) ** 3
        
        # candidates = np.zeros((num_candidates, 3))
        best_F = np.eye(3)
        min_err = np.inf
        for i, (a, b, c) in enumerate(product(coeff_range, repeat=3)):
            F = a * F0 + b * F1 + c * F2 + F3
            # enforce rank 2 cosntraint
            U, S, V_t = np.linalg.svd(F)
            S[-1] = 0
            F = U @ np.diag(S) @ V_t
            # candidates[i] = F
            
            E = self.E_from_F(F)
            trace_constraint = 2 * E @ E.T @ E - np.trace(E @ E.T) * E
            
            if np.linalg.norm(trace_constraint) < 1e-6:
                all_errors = self.sampson_error(pts1, pts2, F)
                err = np.linalg.norm(all_errors)
                if err < min_err:
                    min_err = err
                    best_F = F
                
        return best_F
    
    # def five_point(self, pts1, pts2):
    #     assert pts1.shape[1] == 5 and pts2.shape[1] == 5, 'need exactly 5 points'
    #     T1 = self.normalize(pts1)
    #     T2 = self.normalize(pts2)
    #     pts1_norm = T1 @ pts1
    #     pts2_norm = T2 @ pts2
        
    #     # construct A simialr to 8 point, but 5 points in udner constrained (4 dof null space)
    #     A = np.zeros((5, 9))
    #     for i in range(5):
    #         # homogenous to euclidean
    #         x1, y1 = pts1_norm[:2, i] / pts1_norm[2, i]
    #         x2, y2 = pts2_norm[:2, i] / pts2_norm[2, i]
    #         A[i] = (x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1)
            
    #     _, _, V_t = np.linalg.svd(A)
    #     nullspace = V_t[-4:].T # (9, 4)
    #     F = self.solve_cubic_contraints(pts1, pts2, nullspace)
    
    #     # unnormalize 
    #     F = T2.T @ F @ T1
    #     E = self.E_from_F(F)
    #     return E
    
    def five_point_symbolic(self, pts1, pts2):
        """
        Nistér 5-point algorithm using symbolic elimination via SymPy.
        Inputs:
        pts1, pts2 — 3x5 homogeneous coordinates of corresponding points.
        Returns:
        list of valid 3x3 Essential matrices.
        """
        # 1) Nullspace from 5 points
        A = np.zeros((5, 9))
        for i in range(5):
            x1, y1, w1 = pts1[:, i]
            x2, y2, w2 = pts2[:, i]
            A[i] = [x2*x1, x2*y1, x2*w1,
                    y2*x1, y2*y1, y2*w1,
                    w2*x1, w2*y1, w2*w1]

        # 4D nullspace
        _, _, Vt = np.linalg.svd(A)
        F0, F1, F2, F3 = [Vt[-4 + i].reshape(3, 3) for i in range(4)]

        # 2) Build symbolic variables
        x, y, z = sp.symbols('x y z')

        # Parameterize
        E_sym = (x * sp.Matrix(F0) +
                y * sp.Matrix(F1) +
                z * sp.Matrix(F2) +
                sp.Matrix(F3))

        # Build cubic constraint
        Et = E_sym.T
        term = 2 * E_sym * Et * E_sym - (E_sym * Et).trace() * E_sym
        cubic_eqs = term.reshape(9, 1)

        # Collect all monomials up to cubic
        monoms = []
        degs = [0,1,2,3]
        for dx in degs:
            for dy in degs:
                for dz in degs:
                    if dx + dy + dz <= 3:
                        monoms.append(x**dx * y**dy * z**dz)

        # Build coefficient matrix
        coeffs = []
        for eq in cubic_eqs:
            eq_poly = sp.Poly(eq.expand(), x, y, z)
            coeffs.append([eq_poly.coeffs()[eq_poly.monoms().index(m)] if m in eq_poly.monoms() else 0 for m in monoms])

        M = sp.Matrix(coeffs)

        # Gaussian elimination
        M_red = M.rref()[0]

        # Extract univariate polynomial in z
        poly_z = 0
        for i in range(M_red.rows):
            row = M_red.row(i).tolist()[0]  # convert to list
            poly_z += row[-1] * z**(M_red.cols - 1 - i)

        poly_z = sp.simplify(poly_z)

        # Solve 10th‑degree univariate
        roots = sp.nroots(poly_z)

        real_roots = []
        for r in roots:
            if abs(sp.im(r)) < 1e-8:
                real_roots.append(float(sp.re(r)))

        # Back‑substitute to get x,y
        candidates = []
        for zr in real_roots:
            eqs = []
            for eq in cubic_eqs:
                eqs.append(eq.subs({z: zr}))
            sol_xy = sp.solve(eqs, [x,y], dict=True)

            for sol in sol_xy:
                xr = float(sol[x])
                yr = float(sol[y])
                E_candidate = (xr*F0 + yr*F1 + zr*F2 + F3)

                # Enforce singular values (1,1,0)
                U, S, Vt = np.linalg.svd(E_candidate)
                S_new = np.array([1,1,0])
                E_candidate = U @ np.diag(S_new) @ Vt

                candidates.append(E_candidate)

        # Remove duplicates
        unique_E = []
        for E in candidates:
            if not any(np.allclose(E, U, atol=1e-6) for U in unique_E):
                unique_E.append(E)

        return unique_E
    
    def five_point_ransac(self, pts1, pts2, tol=1.0, max_iterations=1000, min_inliers=0.75):
        N = pts1.shape[1]
        
        max_inliers = -1
        best_inliers_mask = None
        best_E = np.zeros((3, 3))
        for i in range(max_iterations):
            sample_ids = np.random.choice(N, 5, replace=False)
            pts1_sample = pts1[:, sample_ids]
            pts2_sample = pts2[:, sample_ids]

            E = self.five_point_symbolic(pts1_sample, pts2_sample)
            
            error = self.sampson_error(pts1, pts2, E)
            inliers_mask = error < tol
            n_inliers = inliers_mask.sum()
            
            if n_inliers > max_inliers:
                max_inliers = n_inliers
                best_inliers_mask = inliers_mask
                best_E = E
                
            if n_inliers / N >= min_inliers:
                break
        
        return best_E, best_inliers_mask
            
    def _estimate(self, pts1, pts2):
        E, inlier_mask = self.five_point_ransac(pts1, pts2)
        pts1 = pts1[:, inlier_mask]
        pts2 = pts2[:, inlier_mask]
        pose = self.pose_from_E(E, pts1, pts2)
        return pose