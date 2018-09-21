module SVM

    export fit, predict

    mutable struct SVM
        C  # penalty term 惩罚因子
        train_x
        train_y
        w
        b
        gamma
        epsilon  # SVM 目标函数下降的范围
        tol  # KKT 误差
        p::Int  # hyperparameter of polynomial kernel  用于 polynomial 的超参数
        dimension::Int
        count::Int
        kernel_type::String  # kernel: rbf 高斯核函数  linear 线性核函数  polynomial  多项式
        SVM(kernel_type::String, C=10.0, gamma=0.5, p=2) = (x = new(); x.kernel_type = kernel_type; x.C = C; x.gamma = gamma; x.p = p; return x)
    end

    function prepare(m::SVM)
        println("======")
        m.epsilon = 1e-3
        m.tol = 1e-8
        println(m.train_x, " ", m.train_y, " ", m.kernel_type, " ", m.gamma, " ", m.epsilon, " ", m.tol, " ", m.p)
    end

    function fit(m::SVM, x, y)
        m.train_x = x
        m.train_y = y
        init(m)
    end

    function predict(m::SVM, x)

    model = SVM("linear", 2, 0.2)
    prepare(model, 0, 0)
end  # module SVM
