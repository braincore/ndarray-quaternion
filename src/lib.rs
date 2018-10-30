#[macro_use]
extern crate ndarray;
use ndarray::prelude::*;
use std::f32;
use std::ops;

#[derive(Clone, PartialEq, Debug)]
pub struct Quaternion {
    q: Array1<f32>,
}

impl<'a> ops::Mul<Self> for &'a Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: &Quaternion) -> Quaternion {
        Quaternion::new(self.q_matrix().dot(&rhs.q))
    }
}

impl ops::Mul<Self> for Quaternion {
    type Output = Quaternion;

    fn mul(self, rhs: Quaternion) -> Quaternion {
        Quaternion::new(self.q_matrix().dot(&rhs.q))
    }
}

impl Quaternion {
    pub fn new(q: Array1<f32>) -> Self {
        assert_eq!(q.len(), 4);
        Self { q }
    }

    pub fn from_wxyz(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self::new(array![w, x, y, z])
    }

    // v: 3d vector
    pub fn from_vector(v: &Array1<f32>) -> Self {
        assert_eq!(v.len(), 3);
        Self::new(array![0.0, v[0], v[1], v[2]])
    }

    pub fn scalar(&self) -> f32 {
        self.q[0]
    }

    pub fn vector(&self) -> Array1<f32> {
        self.q.slice(s![1..4]).to_owned()
    }

    pub fn to_array(self) -> Array1<f32> {
        self.q
    }

    /// Whether the quaternion is of unit length.
    pub fn is_unit(&self) -> bool {
        (1.0 - self.sum_of_squares()).abs() < f32::EPSILON
    }

    pub fn norm(&self) -> f32 {
        self.sum_of_squares().sqrt()
    }

    pub fn magnitude(&self) -> f32 {
        self.norm()
    }

    fn sum_of_squares(&self) -> f32 {
        self.q.dot(&self.q)
    }

    pub fn normalize(&mut self) {
        if !self.is_unit() {
            let norm = self.norm();
            if norm > 0.0 {
                self.q /= norm
            }
        }
    }

    pub fn normalized(&self) -> Self {
        let mut quat = Self::new(self.q.clone());
        quat.normalize();
        quat
    }

    pub fn unit(&self) -> Self {
        self.normalized()
    }

    pub fn conjugate(&self) -> Self {
        Self::new(array![self.q[0], -self.q[1], -self.q[2], -self.q[3]])
    }

    pub fn inverse(&self) -> Self {
        let s = self.sum_of_squares();
        assert!(s > 0.0);
        let mut conj = self.conjugate();
        conj.q /= s;
        conj
    }

    fn q_matrix(&self) -> Array2<f32> {
        array![
            [self.q[0], -self.q[1], -self.q[2], -self.q[3]],
            [self.q[1], self.q[0], -self.q[3], self.q[2]],
            [self.q[2], self.q[3], self.q[0], -self.q[1]],
            [self.q[3], -self.q[2], self.q[1], self.q[0]],
        ]
    }

    pub fn to_vector(&self) -> Array1<f32> {
        array![self.q[1], self.q[2], self.q[3],]
    }

    pub fn rotate_vector(&self, v: &Array1<f32>) -> Array1<f32> {
        let qv = Self::from_vector(&v);
        let rotated = self * &qv * self.conjugate();
        rotated.to_vector()
    }

    /// Returns [yaw, pitch, roll]
    pub fn taitbryan(&self) -> Array1<f32> {
        let q = &self.q;
        let tb1 = 2.0 * (q[0] * q[2] - q[1] * q[3]).asin();
        let tb0 =
            (2.0 * (q[2] * q[3] + q[0] * q[1])).atan2(1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]));
        let tb2 =
            (2.0 * (q[1] * q[2] + q[0] * q[3])).atan2(1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
        array![tb0, tb1, tb2]
    }
}

#[cfg(test)]
mod tests {
    use super::Quaternion;

    #[test]
    fn is_unit() {
        let q = Quaternion::new(array![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(q.is_unit(), true);

        let q = Quaternion::new(array![1.0, 0.5, 0.5, 0.5]);
        assert_eq!(q.is_unit(), false);
    }

    #[test]
    fn sum_of_squares() {
        let q = Quaternion::new(array![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(q.sum_of_squares(), 1.0);

        let q = Quaternion::new(array![1.0, 0.5, 0.5, 0.5]);
        assert_eq!(q.sum_of_squares(), 1.75);
    }

    #[test]
    fn scalar_vector() {
        let q = Quaternion::new(array![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(q.scalar(), 0.1);
        assert_eq!(q.vector(), array![0.2, 0.3, 0.4]);
    }

    #[test]
    fn norm() {
        let q = Quaternion::new(array![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(q.norm(), 1.0);

        let q = Quaternion::new(array![1.0, 0.5, 0.5, 0.5]);
        assert_eq!(q.norm(), 1.3228756555322954);
    }

    #[test]
    fn normalize() {
        let mut q = Quaternion::new(array![0.5, 0.5, 0.5, 0.5]);
        q.normalize();
        assert_eq!(q.norm(), 1.0);

        let mut q = Quaternion::new(array![1.0, 0.5, 0.5, 0.5]);
        q.normalize();
        // Unfortunately, difference is larger than f32::EPSILON.
        assert!((1.0 - q.norm()).abs() < 0.00001);

        let mut q = Quaternion::new(array![0.0, 0.0, 0.0, 0.0]);
        q.normalize();
        assert_eq!(q.norm(), 0.0);
    }

    #[test]
    fn conjugate() {
        let q = Quaternion::new(array![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(q.conjugate().q, array![0.5, -0.5, -0.5, -0.5]);
    }

    #[test]
    fn eq() {
        let q = Quaternion::new(array![0.1, 0.2, 0.3, 0.4]);
        let q2 = q.clone();
        assert_eq!(q, q2);
    }

    #[test]
    fn inverse() {
        let q = Quaternion::new(array![-0.754, -0.18, -0.327, 0.54]);
        assert_eq!(
            q.inverse().q,
            array![-0.75441870, 0.18009995, 0.32718155, -0.54029983]
        );
    }

    #[test]
    fn rotate() {
        let v = array![-7.135, -0.297, 6.37];
        let q = Quaternion::new(array![-0.754, -0.18, -0.327, 0.54]);
        let v2 = q.rotate_vector(&v);
        assert_eq!(v2, array![0.18197358, 0.8871603, 9.521111])
    }
}
