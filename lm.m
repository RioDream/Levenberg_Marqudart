
%本例中目标函数是 sum||obs_1(i)-f||
syms a b y x real;

%需要拟合的函数f的形式，其值后面常用y表示
f = a*exp(-b*x);
Jsym = jacobian(f,[a,b]);

%拟合用数据。参见《数学实验》，P190，例2
%data_1是x的值，obs_1是函数f值
data_1 = [0.25,   0.5,    1,  1.5,     2,   3,   4,    6,    8];
obs_1  = [19.21 18.15 15.36 14.10  12.89 9.32 7.45  5.24  3.01];

%2.LM算法
%初始猜测s
a0 = 10;
b0 = 0.5;
y_init = a0*exp(b0*data_1);

%数据个数
Ndata= length(obs_1);
%参数维数 参数只有两个 a,b
Nparams = 2;
%迭代最大次数
n_iters = 50;
%LM算法的阻尼系数初值
lamda = 0.01;

%step 1:变量赋值
updateJ = 1;%是否更新J
a_est = a0;
b_est = b0;

%step 2:迭代
for it = 1:n_iters
	if updateJ==1
		%根据当前估计值，计算出雅克比矩阵
		J = zeros(Ndata,Nparams);
		for i=1:length(data_1)
			%逐行求取导数
			J(i,:) = [exp(-b_est*data_1(i)) - a_est*data_1(i)*exp(-b_est*data_1(i))];
		end

		%根据当前参数，得到函数值
		y_est = a_est*exp(-b_est*data_1);
		%计算误差
		d = obs_1 - y_est;
		%计算海森矩阵的近似
		H = J'*J;
		%若是第一次迭代
		if it==1
			e = dot(d,d);
		end
	end

	%根据阻尼系数lamda混合得到H矩阵
	H_lm = H+lamda*eye(Nparams);

	%计算步长 dp
	dp = inv(H_lm)*(J'*d(:));
	g = J'*d(:);
	a_lm = a_est + dp(1);
	b_lm = b_est + dp(2);

	%计算新的可能估计值，对应的y和相关残差e
	y_est_lm = a_lm * exp(-b_lm*data_1);
	d_lm = obs_1-y_est_lm;
	e_lm = dot(d_lm,d_lm);

	%根据误差，决定如何更新参数和阻尼系数
	if e_lm<e
		%如果误差变小了，减少阻尼系数，更接近高斯牛顿法，并继续迭代
		lamda = lamda/10;
		a_est = a_lm;
		b_est = b_lm;
		e = e_lm;
		disp(e);
		updateJ = 1;
	else
		%如果误差变大了，就增大阻尼系数，更接近梯度下降法
		updateJ = 0;
		lamda = lamda*10;
	end
end
