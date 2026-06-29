# Quantum-MPS

python encoder_nsl_kdd_full_binary.py ./nsl_kdd

python train_mps_nsl_kdd.py ./nsl_kdd/

python mps_feature_elimination_order.py ./nsl_kdd/ --no-train

python log_backend_target.py --backend fake:fez --out ./nsl_kdd/backend_target_fake_fez.json

python sweep_models.py ./nsl_kdd --feature-order ./nsl_kdd/order.json --k-list 4,6,8,10,12,14 --bond-list 4 --variants no_reuse_isometry --fixed-bond --seeds 0,1,2,3,4 --device fez --summary-mode overwrite

python sweep_models.py ./nsl_kdd --feature-order ./nsl_kdd/order.json --k-list 8 --bond-list 2,4,8 --variants no_reuse_isometry --fixed-bond --seeds 0,1,2,3,4 --device fez

python sweep_models.py ./nsl_kdd --feature-order ./nsl_kdd/order.json --k-list 8 --bond-list 4 --variants no_reuse_isometry,reuse_unitary,reuse_isometry,no_reuse_unitary --fixed-bond --seeds 0,1,2,3,4 --device fez

python sweep_models.py ./nsl_kdd --feature-order ./nsl_kdd/order.json --k-list 14 --bond-list 4 --variants no_reuse_isometry,reuse_unitary,reuse_isometry,no_reuse_unitary --fixed-bond --seeds 0,1,2,3,4 --device fez

python sweep_models.py ./nsl_kdd --feature-order ./nsl_kdd/order.json --k-list 8 --bond-list 8 --variants no_reuse_isometry,reuse_unitary,reuse_isometry,no_reuse_unitary --fixed-bond --seeds 0,1,2,3,4 --device fez
