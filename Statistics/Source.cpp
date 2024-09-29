#include <iostream>
#include <functional>
#include <array>
#include <ranges>
#include <algorithm>
#include <utility>
#include <string>
#include <unordered_set>
#include <numbers>
#include <deque>
#include <numeric>
#include <map>
#include <random>
#include <span>
#include <string_view>
#include <iomanip>
//List of Common Probability Distributions
//We recently published an article on our website that we thought you might find helpful, particularly if you are studying for an exam.This article can serve as a useful reference tool for a quick consultation.We hope you find it useful.
//
//List of Common Probability Distributions
//
//Several probability distributions are used in statistics.These include the three most commonly used distributions : the Normal Distribution, the Binomial Distribution, and the Poisson Distribution.
//
//Three Most Commonly Used Distributions :
//
//1. Normal Distribution : The normal distribution is a continuous probability distribution that is symmetrical around the mean.It is often used to model normally distributed data, such as height or intelligence.The normal distribution is defined by two parameters : the mean(the average value of the data) and the standard deviation, which measures the spread of the data around the mean.
//
//2. Binomial Distribution : The binomial distribution is a discrete probability distribution that models the probability of a specific number of successes in a fixed number of independent trials.It is often used to model the probability of success in experiments with a fixed number of trials and two possible outcomes, such as the probability of heads in a series of coin flips.The binomial distribution is defined by the probability of success in each trial and the number of trials.
//
//3. Poisson Distribution : The Poisson distribution is a discrete probability distribution that models the probability of a specific number of events occurring in a fixed time interval.It is often used to model the probability of a specific number of occurrences of a rare event, such as the number of accidents at an intersection.The Poisson distribution is defined by a single parameter, the average number of occurrences of the event in the given time interval.
//
//
//
//Other Distributions :
//
//
//
//In addition to the three distributions listed above, various other distributions are frequently used under different circumstances.
//4. Uniform Distribution : The uniform distribution is a continuous probability distribution that is constant over a given range.It is often used to model uniformly distributed data, such as in random number generation.The uniform distribution is defined by two parameters : the lower and upper bounds of the range.
//
//5. Bernoulli Distribution : The Bernoulli distribution is used when there are only two possible outcomes, such as a coin flip or a yes / no question.It is also known as the binary distribution and can be used to model the probability of success in a single trial.
//
//6. Geometric Distribution : The geometric distribution is a discrete probability distribution that models the number of failures before the first success in a sequence of independent trials.It is often used to model the number of times an event must be tried before success, such as the number of coin flips required to get heads.The geometric distribution is defined by a single parameter : the probability of success in each trial.
//
//7. Negative Binomial Distribution : The negative binomial distribution is a discrete probability distribution similar to the binomial distribution.Instead of modelling the number of successes in a fixed number of trials, it models the number of failures before a fixed number of successes.It is often used to model the number of failures before a target number of successes is reached, such as the number of losses before a sports team wins a certain number of games.The negative binomial distribution is defined by two parameters : the probability of success in each trial and the target number of successes.
//
//8. Hypergeometric Distribution : Hypergeometric distribution is a discrete probability distribution that models the probability of a specific number of successes in a sample drawn from a finite population without replacement.It is often used to model the probability of a certain number of successes in a sample drawn from a population with a known number of successes and failures, such as the probability of drawing a certain number of red balls from a jar with a known number of red and blue balls.The hypergeometric distribution is defined by three parameters : the population size, the number of successes in the population, and the sample size.
//
//9. Exponential Distribution : The exponential distribution is a continuous probability distribution that models the time between events in a Poisson process.It is often used to model the time between occurrences of a rare event, such as the time between failures of a piece of equipment.The exponential distribution is defined by a single parameter : the rate at which events occur.
//
//10. Log - Normal Distribution : The log - normal distribution is a continuous probability distribution that is often used to model data that is skewed to the right and has a long tail.It is defined by two parameters : the mean and standard deviation of the underlying normal distribution.
//
//11. Weibull Distribution : The Weibull distribution is a continuous probability distribution that is often used to model data with a minimum and maximum value and is either positively or negatively skewed.It is defined by three parameters : the shape parameter, which determines the shape of the distribution; the scale parameter, which determines the location and scale of the distribution; and the threshold parameter, which determines the point at which the distribution starts.The Weibull distribution is often used to model data that represents the time to failure of a system or the time to occurrence of an event.It is also used to model data representing a product's or system's lifetime or lifespan.
//
//12. Chi - Square Distribution : The chi - squared distribution is a continuous probability distribution that is often used in hypothesis testing, particularly in tests of goodness of fit, as well as to test the independence of two variables in a contingency table.It is defined by a single parameter : the number of degrees of freedom, which determines the shape of the distribution.
//
//13. Student’s t - Distribution : The t - distribution is a continuous probability distribution often used to model normally distributed data but with unknown variance and a small sample size.It is used in hypothesis testing and confidence interval estimation and is defined by a single parameter : the degrees of freedom, which determines the shape of the distribution.The t - distribution has a variety of shapes depending on the number of degrees of freedom, ranging from a symmetric bell - shaped curve when the number of degrees of freedom is large to a skewed distribution when the number of degrees of freedom is small.The t - distribution is also known as the Student's t-distribution, named after William Gosset, who used it to analyze small samples in the early 20th century.
//
//14. F - Distribution : The F - distribution is a continuous probability distribution that is often used to compare the variances of two samples.It is used in hypothesis testing and is defined by two parameters : the numerator degrees of freedom and the denominator degrees of freedom, which determine the shape of the distribution.
//
//These are just a few examples of other probability distributions that are used in statistics.There are many others, each with its own specific uses and characteristics.

template <typename Type = size_t>
class MATRIX
{
	using VECTOR = std::vector<Type>;
public:
	MATRIX() = default;
	explicit MATRIX(size_t rows, size_t columns, const std::string& rowNames = "", const std::string& columnNames = "");
	MATRIX(const MATRIX& matrix);
	MATRIX(const std::vector<VECTOR>& vectors);
	template <typename OtherType>
	MATRIX(const MATRIX<OtherType>& other);
	const VECTOR& operator[](uint32_t index) const;
	VECTOR& operator[](uint32_t index);

	bool operator==(const MATRIX& other) const;

	MATRIX operator-() const;

	MATRIX GetNthDegree(size_t degree);

	friend MATRIX operator+(const MATRIX& left, const MATRIX& right);
	friend MATRIX operator-(const MATRIX& left, const MATRIX& right);
	template<typename Type>
	friend MATRIX<Type> operator*(const MATRIX<Type>& left, const MATRIX<Type>& right);
	friend MATRIX operator*(double left, const MATRIX& right);
	friend MATRIX operator*(const MATRIX& left, double right);
	friend MATRIX operator/(const MATRIX& left, double right);

	MATRIX& operator+=(const MATRIX& other);
	MATRIX& operator-=(const MATRIX& other);
	MATRIX& operator*=(const MATRIX& other);
	MATRIX& operator*=(double other);
	MATRIX& operator/=(double other);
	MATRIX& operator-=(double other);
	MATRIX& operator+=(double other);

	inline size_t GetRows() const { return matrix.size(); }
	inline size_t GetColumns() const { return (*matrix.begin()).size(); }

	VECTOR GetColumn(size_t index) const;
	VECTOR GetRow(size_t index) const;

	double GetDeterminant() const;

	void PrintMatrix() const;

	MATRIX GetTransposed() const;
	static double SmallMatrixLaibnizDeterminant(const MATRIX& type);
	static double CalculateAndDivideMatrices(const MATRIX& type);
	MATRIX CreateMatrixWithoutColumn(size_t column) const;
	inline void SetColumnName(const std::string& columnName) { this->columnName = columnName; };
	inline void SetRowName(const std::string& rowName) { this->rowName = rowName; };
private:
	std::vector<VECTOR> matrix;
	std::string columnName = "column";
	std::string rowName = "row";
};

namespace statistics
{
	template <typename VecType>
	VecType::value_type sum(const VecType& vec, std::function<bool(typename VecType::value_type)> ifParam = nullptr)
	{
		typename VecType::value_type s{ 0 };
		for (size_t elem{ 0u }; elem < std::size(vec); ++elem)
		{
			if (ifParam)
			{
				if (ifParam(vec[elem]))
					s += vec[elem];
			}
			else
				s += vec[elem];
		}
		return s;
	}

	template <typename VecType> 
	auto avg(const VecType& vec, std::function<bool(typename VecType::value_type)> ifParam = nullptr)
	{
		typename VecType::value_type s{ 0 };
		size_t passed = ifParam ? std::ranges::count_if(vec, ifParam) : std::size(vec);
		for (size_t elem{ 0u }; elem < std::size(vec); ++elem)
		{
			if (ifParam)
			{
				if (ifParam(vec[elem]))
					s += vec[elem];
			}
			else
				s += vec[elem];
		}

		return static_cast<float>(s) / passed;
	}

	template <typename VecType>
	auto mean(const VecType& vec)
	{
		const size_t size = std::size(vec);

		if (size > 0)
		{
			VecType copy = vec;
			std::ranges::sort(copy);
			if (size % 2 == 1)
				return static_cast<float>(copy[size / 2]);
			else
				return static_cast<float>(copy[size / 2] + copy[(size / 2) - 1]) / 2.0f;
		}

		return 0.0f;
	}

	template <typename VecType>
	auto sumOfSquares(const VecType& vec)
	{
		auto result = 0.0;
		const auto avg = statistics::avg(vec);
		for (auto elem{ 0u }; elem < std::size(vec); ++elem)
			result += static_cast<float>(std::pow(vec[elem] - avg, 2));

		return result;
	}

	template <typename VecType>
	auto dev(const VecType& vec, bool isPopulation = true)
	{
		const auto size = std::size(vec);
		if (size > 1)
		{
			auto result = sumOfSquares(vec);
			return std::sqrt(result / (isPopulation ? size : (size - 1)));
		}

		return 0.0f;
	}

	template <typename VecType>
	auto var(const VecType& type, bool isPopulation = true)
	{
		return std::pow(dev(type, isPopulation), 2);
	}

	template <typename VecType>
	float range(const VecType& type)
	{
		const auto size = std::size(type);
		if (size > 1)
		{
			auto min = std::numeric_limits<typename VecType::value_type>::max();
			auto max = std::numeric_limits<typename VecType::value_type>::min();
			for (const auto& elem : type)
			{
				if (elem >= max)
					max = elem;
				if (elem <= min)
					min = elem;
			}

			return static_cast<float>(max) - static_cast<float>(min);
		}

		return 0.0f;
	}

	template <typename VecType>
	auto max(const VecType& vec)
	{
		auto elem = std::numeric_limits<typename VecType::value_type>::min();
		for (const auto& e : vec)
			if (e >= elem)
				elem = e;

		return elem;
	}

	template <typename VecType>
	auto min(const VecType& vec)
	{
		auto elem = std::numeric_limits<typename VecType::value_type>::max();
		for (const auto& e : vec)
			if (e <= elem)
				elem = e;

		return elem;
	}

	template<typename VecType>
	auto mode(const VecType& vec)
	{
		std::vector<typename VecType::value_type> modes;
		std::unordered_map<typename VecType::value_type, size_t> mode_elems;
		for (const auto& e : vec)
			mode_elems[e]++;

		// simplify this with new standard
		
		const auto maxFound = std::ranges::max(std::ranges::views::values(mode_elems));
		for (const auto& [key, value] : mode_elems)
			if (value == maxFound)
				modes.push_back(key);

		return modes;
	}

	template <typename VecType>
	auto stderror(const VecType& vec)
	{
		const auto size = std::size(vec);
		if (size > 1)
		{
			return statistics::dev(vec, false) / size;
		}

		return 0.0f;
	}

	template <typename VecType>
	auto skew1(const VecType& vec)
	{
		const auto size = std::size(vec);

		if (size > 2)
		{
			auto left = static_cast<float>(size) / ((size - 1) * (size - 2));
			auto right = 0.0f;
			const auto avg = statistics::avg(vec);
			const auto dev = statistics::dev(vec);
			for (const auto& elem : vec)
			{
				right += std::powf((elem - avg) / dev, 3);
			}

			return left * right;
		}
		return 0.0f;
	}

	template <typename VecType>
	auto skew2(const VecType& vec)
	{
		const auto size = std::size(vec);
		if (size > 1)
		{
			float up = 0.0f;
			const auto avg = statistics::avg(vec);
			float down = 0.0f;
			for (const auto& elem : vec)
			{
				up += std::powf((elem - avg), 3);
				down += std::powf((elem - avg), 2);
			}

			up = up / size;
			down = std::powf((down / size), 1.5f);
			return up / down;
		}

		return 0.0f;
	}

	template <typename VecType>
	auto kurtosis1(const VecType& vec)
	{
		const auto size = std::size(vec);

		if (size > 2)
		{
			auto left = static_cast<float>((size) * (size + 1)) / ((size - 1) * (size - 2) * (size - 3));
			auto middle = 0.0f;
			const auto avg = statistics::avg(vec);
			const auto dev = statistics::dev(vec);
			for (const auto& elem : vec)
			{
				middle += std::powf((elem - avg) / dev, 4);
			}

			float right = static_cast<float>(3.0f * std::pow(size - 1, 2.0f)) /  ((size - 2) * (size - 3));
			return left * middle - right;
		}
		return 0.0f;
	}

	template <typename VecType>
	auto kurtosis2(const VecType& vec)
	{
		const auto size = std::size(vec);
		if (size > 1)
		{
			float up = 0.0f;
			const auto avg = statistics::avg(vec);
			float down = 0.0f;
			for (const auto& elem : vec)
			{
				up += std::powf((elem - avg), 4);
				down += std::powf((elem - avg), 2);
			}

			up = up / size;
			down = std::powf((down / size), 2);
			return up / down;
		}

		return 0.0f;
	}

	template<typename VectData, typename Labels>
	void DrawHistogram(const VectData& data, const Labels& labels)
	{
		if (std::size(data) != std::size(labels) || std::size(data) == 0)
			return;

		for (size_t i{ 0u }; i < std::size(data); ++i)
		{
			std::cout << labels[i] << " | ";
			const auto dots = static_cast<size_t>(data[i]);
			for (size_t j{ 0u }; j < dots; ++j)
				std::cout << ".";

			std::cout << std::endl;
		}
	}

	template<typename VectData, typename Limits>
	void DrawBonsHistogram(const VectData& data, const Limits& limits)
	{
		std::vector<size_t> limitsVec(std::size(limits) + 1);
		std::deque<typename Limits::value_type> limitsFull;
		for (const auto& l : limits)
			limitsFull.push_back(l);

		limitsFull.push_front(std::numeric_limits<typename Limits::value_type>::min());
		limitsFull.push_back(std::numeric_limits<typename Limits::value_type>::max());

		for (const auto& elem : data)
		{
			for (size_t i{ 0u }; i < limitsFull.size() - 1; ++i)
			{
				if (elem > limitsFull[i] && elem <= limitsFull[i + 1])
					limitsVec[i]++;
			}
		}
		std::cout << "-INF\n";
		std::cout << "\t |";
		for (size_t i{ 0u }; i < limitsVec[0]; ++i)
		{
			std::cout << "|";
		}
		std::cout << "\n";
		for (size_t i{ 1u }; i < std::size(limits) - 1; ++i)
		{
			std::cout << limits[i] << "\n";
			std::cout << "\t |";
			for (size_t j{ 0u }; j < limitsVec[i]; ++j)
			{
				std::cout << "|";
			}
			std::cout << "\n";
		}
		std::cout << limits[std::size(limits) - 1] << "\n";
		std::cout << "\t |";
		for (size_t i{ 0u }; i < limitsVec.back(); i++)
		{
			std::cout << "|";
		}
		std::cout << "\n";
		std::cout << "+INF";
		std::cout << "\n";
	}

	constexpr auto GridSize = 50u;

	template <typename VecTypeX, typename VecTypeY> 
	void DrawCoordinate(const VecTypeX& x, const VecTypeY& y)
	{
		VecTypeX xCopy = x;
		VecTypeY yCopy = y;
		
		std::vector<std::vector<bool>> Matrix(GridSize, std::vector<bool>(GridSize, false));

		for (size_t val{ 0u }; val < std::size(x); ++val)
		{/*
			xCopy[val] = std::clamp(xCopy[val], -25, 25);
			yCopy[val] = std::clamp(yCopy[val], -25, 25);*/
			if (xCopy[val] > 24)
				xCopy[val] = 24;
			else if (xCopy[val] < -24)
				xCopy[val] = -24;

			if (yCopy[val] > 24)
				yCopy[val] = 24;
			else if (yCopy[val] < -24)
				yCopy[val] = -24;


			yCopy[val] *= -1;
			xCopy[val] = (xCopy[val] + 24);
			yCopy[val] = (yCopy[val] + 24);
			Matrix[static_cast<size_t>(xCopy[val])][static_cast<size_t>(yCopy[val])] = true;
		}

		for (size_t i = 0; i < GridSize; ++i)
		{
			for (size_t j = 0; j < GridSize; ++j)
			{
				if (Matrix[j][i])
				{
					std::cout << "o";
				}
				else if (i == GridSize / 2 && j == GridSize / 2)
				{
					std::cout << "+";
				}
				else if (i == (GridSize / 2))
				{
					std::cout << "-";
				}
				else if (j == GridSize / 2)
				{
					std::cout << "|";
				}
				if (i != GridSize / 2)
					std::cout << " ";
			}
			std::cout << "\n";
		}
	}

	template <typename VecType>
	auto CreateBins(const VecType& vec)
	{
		std::vector<float> bins;
		const auto size = std::size(vec);
		const float rangeVec = range(vec);
		const auto numElems = static_cast<size_t>(std::sqrt(size));
		const float step = rangeVec / numElems;
		float beg = min(vec);
		bins.push_back(beg);
		float end = max(vec);
		while (beg < end)
		{
			beg += step;
			bins.push_back(beg);
		}
		return bins;
	}

	// doesnt work for big numbers
	// but it is fine for test purpose 
	unsigned long long fact(size_t number)
	{
		unsigned long long result = 1llu;
		for (size_t i{ 2u }; i <= number; ++i)
			result *= i;

		return result;
	}

	unsigned long long perm(size_t repetion, size_t number_choose, bool repetition = false)
	{
		if (repetition)
		{
			return static_cast<unsigned long long>(std::pow(number_choose, repetion));
		}
		else
		{
			return fact(repetion) / fact(repetion - number_choose);
		}
	};

	unsigned long long comb(size_t draws, size_t combinations)
	{
		return fact(combinations) / (fact(combinations - draws) * fact(draws));
	}

	constexpr auto steps = 100'000ul;
	double TrapezoidalRule(double lower, double upper, std::function<double(double)> f)
	{
		const double h = (upper - lower) / steps;
		double sum = f(lower) + f(upper);
		for (auto i{ 0u }; i < steps; ++i)
		{
			sum += 2 * f(lower + i * h);
		}

		return (h / 2) * sum;
	}

	double SimpsonsRule(double lower, double upper, std::function<double(double)> f)
	{
		const double h = (upper - lower) / steps;
		double sum = f(upper) + f(lower);
		for (auto i = 1u; i < steps; ++i)
		{
			if (i % 2 == 0)
			{
				sum += 2 * f(lower + i * h);
			}
			else
			{
				sum += 4 * f(lower + i * h);
			}
		}

		return (h / 3) * sum;
	}

	template <typename VecType>
	auto CalculateSumOfSquaresWithin(const std::vector<VecType>& samples) // error
	{
		double result = 0.0;
		std::ranges::for_each(samples, [&result](const auto& sample) { result += sumOfSquares(sample); });
		return result;
	}

	template <typename VecType>
	auto CalculateSumOfSquaresBetween(const std::vector<VecType>& samples)
	{
		double result = 0.0;
		std::vector<double> averages;
		std::ranges::for_each(samples, [&averages](const auto& sample) 
			{
				averages.emplace_back(statistics::avg(sample));
			});
		const double averageOfAverages = (avg(averages));
		for (auto sampleIt{ 0u }; sampleIt < std::size(averages); ++sampleIt)
		{
			result += std::pow((averages[sampleIt] - averageOfAverages), 2) * std::size(samples[sampleIt]);
		}
		
		return result;
	}


	template <typename VecType>
	auto SST(const std::vector<VecType>& samples) // Sum of squares total
	{
		return CalculateSumOfSquaresBetween(samples) + CalculateSumOfSquaresWithin(samples);
	}

	template <typename VecType>
	size_t DegreesOFFreedomBeetwen(const std::vector<VecType>& samples)
	{
		return std::size(samples) - 1;
	}

	template <typename VecType>
	size_t DegreesOFFreedomWithin(const std::vector<VecType>& samples)
	{
		size_t result{ 0u };
		std::ranges::for_each(samples, [&result](const auto& sample) {result += std::size(sample); });
		return result - std::size(samples);
	}

	template <typename VecType>
	size_t DegreesOFFreedomTotal(const std::vector<VecType>& samples)
	{
		size_t result{ 0u };
		std::ranges::for_each(samples, [&result](const auto& sample) {result += std::size(sample); });
		return result - 1;
	}

	template <typename VecType>
	auto MeanSSBetween(const std::vector<VecType>& samples)
	{
		return CalculateSumOfSquaresBetween(samples) / DegreesOFFreedomBeetwen(samples);
	}

	template <typename VecType>
	auto MeanSSWithin(const std::vector<VecType>& samples)
	{
		return CalculateSumOfSquaresWithin(samples) / DegreesOFFreedomWithin(samples);
	}

	template <typename VecType>
	auto CalculateFRationValue(const std::vector<VecType>& samples)
	{
		return MeanSSBetween(samples) / MeanSSWithin(samples);
	}

	double BetaFunction(double x, double y)
	{
		return (std::tgamma(x) * std::tgamma(y)) / std::tgamma(x + y);
	}

	double fCDF(double x, size_t df1, size_t df2)
	{
		double num = std::pow(df1 * x, df1) * std::pow(df2, df2);
		double denom = std::pow(df1 * x + df2, df1 + df2);
		return num / denom * BetaFunction(df1 / 2.0, df2 / 2.0);
	}

	double inverseF(double alpha, size_t df1, size_t df2, double epsilon = 1e-6) 
	{
		// TODO: NOT GOOD!!!
		// Simple binary search to find the critical F-value
		double low = 0.0, high = 10.0, mid;
		while (high - low > epsilon) 
		{
			mid = (low + high) / 2.0;
			if (fCDF(mid, df1, df2) < alpha) 
			{
				low = mid;  // Move the lower bound up
			}
			else 
			{
				high = mid; // Move the upper bound down
			}
		}
		return mid;
	}

	template <typename VecType>
	auto ANOVA(const std::vector<VecType>& samples)
	{
		constexpr auto alpha = 0.05f;
		const auto df1 = DegreesOFFreedomBeetwen(samples);
		const auto df2 = DegreesOFFreedomWithin(samples);
		const auto criticalFValue = inverseF(alpha, df1, df2);
		const auto testFValue = CalculateFRationValue(samples);
		if (testFValue > criticalFValue)
		{
			std::cout << "Reject null hypothesis" << std::endl;
		}
		else
		{
			std::cout << "Accept null hypothesys" << std::endl;
		}
	}

	double getCriticalChiSquared(double alpha, size_t degreesOfFreedom)
	{
		std::map<std::pair<size_t, double>, double> chiSquaredTable = 
		{
			{{1, 0.05}, 3.8415}, {{1, 0.01}, 6.6349},
			{{2, 0.05}, 5.9915}, {{2, 0.01}, 9.2103},
			{{3, 0.05}, 7.8147}, {{3, 0.01}, 11.3449},
			{{4, 0.05}, 9.4877}, {{4, 0.01}, 13.2767},
			{{5, 0.05}, 11.0705}, {{5, 0.01}, 15.0863},
		};

		auto key = std::make_pair(degreesOfFreedom, alpha);
		if (chiSquaredTable.find(key) != chiSquaredTable.end()) 
		{
			return chiSquaredTable[key];
		}
		else
		{
			return -1.0;
		}
	}

	// testing if sample follow the distribution
	template <typename VecType>
	auto CalculatedStatisticsFitTest(const std::vector<VecType>& samplesObserved, const std::vector<VecType>& samplesExpected)
	{
		double result = 0.0;
		for (size_t elem{ 0u }; elem < samplesObserved.size(); ++elem)
		{
			result += std::pow(samplesObserved[elem] - samplesExpected[elem], 2.0) / samplesExpected[elem];
		}
		return result;
	}


	template <typename VecType>
	void GoodnessOfFitTest(const std::vector<VecType>& samplesObserved, const std::vector<VecType>& samplesExpected)
	{
		const auto calculatedFitTest = CalculatedStatisticsFitTest(samplesObserved, samplesExpected);
		if (auto criticalValue = getCriticalChiSquared(0.05, std::size(samplesExpected) - 1); calculatedFitTest > criticalValue)
		{
			std::cout << "Null hypothesis rejected" << std::endl;
		}
		else
		{
			std::cout << "Accepted" << std::endl;
		}
	}

	template <typename Type>
	void ContigencyTable(const MATRIX<Type>& table)
	{
		double resultSum = 0.0;
		for (uint32_t i{ 0u }; i < table.GetRows(); ++i)
		{
			for (uint32_t j{ 0u }; j < table.GetColumns(); ++j)
			{
				resultSum += table[i][j];
			}
		}

		std::vector<Type> sumRows, sumColumns;
		for (size_t i = 0; i < table.GetRows(); ++i)
			sumRows.emplace_back(statistics::sum(table.GetRow(i)));

		for (size_t i = 0; i < table.GetColumns(); ++i)
			sumColumns.emplace_back(statistics::sum(table.GetColumn(i)));

		MATRIX<double> expected(table.GetRows(), table.GetColumns());
		double expetedValueSum = 0.0f;
		for (uint32_t i{}; i < expected.GetRows(); ++i)
		{
			for (uint32_t j{}; j < expected.GetColumns(); ++j)
			{
				expected[i][j] = static_cast<double>(sumRows[i] * sumColumns[j] / resultSum);
				expetedValueSum += expected[i][j];
			}
		}

		double sumAll = 0.0f;
		for (uint32_t row{}; row < expected.GetRows(); ++row)
		{
			for (uint32_t column{}; column < expected.GetColumns(); ++column)
			{
				sumAll += std::pow(static_cast<double>(table[row][column]) - expected[row][column], 2) / expected[row][column];
			}
		}

		auto criticalStatistics = getCriticalChiSquared(0.05, (expected.GetRows() - 1) * (expected.GetColumns() - 1));
		if (sumAll > criticalStatistics)
		{
			std::cout << "Null hypothesis rejected, there is dependency!!!" << std::endl;
		}
		else
		{
			std::cout << "Null hypothesis accepted!!! " << std::endl;
		}
	}

	template <typename VecType>
	double CorrelationFactorR(const VecType& x, const VecType& y)
	{
		const auto xAvg = avg(x);
		const auto yAvg = avg(y);

		double upperPart = 0.0;

		for (size_t i{}; i < std::size(x); ++i)
		{
			upperPart += ((x[i] - xAvg) * (y[i] - yAvg));
		}

		double downPartLeft = 0.0, downPartRight = 0.0;
		for (size_t i{}; i < std::size(x); ++i)
		{
			downPartLeft += std::pow(x[i] - xAvg, 2);
			downPartRight += std::pow(y[i] - yAvg, 2);
		}

		return upperPart / (std::sqrt(downPartLeft * downPartRight));
	}

	void CheckIsThereCorrelation(const double correlationFactorR)
	{
		if (correlationFactorR < std::numeric_limits<double>::epsilon())
		{
			std::cout << "Not correlated! \n" << std::endl;
		}
		else if (correlationFactorR < -0.5 || correlationFactorR > 0.5)
		{
			std::cout << "Strong correlated! \n";
		}
		else
		{
			std::cout << "Weak correlated! \n";
		}
	}

	void PercentageOfCorrelation(const double correlationR)
	{
		std::cout << static_cast<size_t>(std::pow(correlationR, 2) * 100) << "% correlation is because of Xes \n";
	}

    // Regression
	// Gets a and b from y = a + bx
	template <typename VecType> 
	std::pair<double, double> CalculateABRegression(const VecType& x, const VecType& y)
	{
		const auto N = std::size(x);
		const auto sumX = static_cast<double>(sum(x));
		const auto sumY = static_cast<double>(sum(y));
		double upL{}, upR{sumX * sumY}, downL{}, downR{std::pow(sumX, 2)};
		for (size_t i{}; i < N; ++i)
		{
			upL += (x[i] * y[i]);
			downL += std::pow(x[i], 2);
		}
		const double b = (N * upL - upR) / (N * downL - downR);
		const double a = (sumY - b * sumX) / N;
		return std::make_pair(a, b);
	}

	template <typename VecType>
	double GetPredictionOFY(const VecType& x, const VecType& y, const double XTest)
	{
		const auto [a, b] = CalculateABRegression(x, y);
		return a + b * XTest;
	}


	class Random
	{
	public:
		Random() = default;
		long long GetUniformInt(long bottom, long top)
		{
			std::uniform_int_distribution<long long> dist(bottom, top);
			return dist(engine);
		}

		float GetUniformFloat(float bottom, float top)
		{
			std::uniform_real_distribution<float> dist(bottom, top);
			return dist(engine);
		}

		float GetNormalDistribution(float mean, float sigma)
		{
			std::normal_distribution<float> dist(mean, sigma);
			return dist(engine);
		}
		size_t GetBinomialDist(float p)
		{
			std::binomial_distribution<size_t> dist(1, p);
			return dist(engine);
		}

		double NormalDistributionDensity(const float x, const float sigma, const float mean)
		{
			const auto sigmaSquare = std::pow(sigma, 2.0);
			const auto exp = -(std::pow(x - mean, 2) / 2 * sigmaSquare);
			return std::pow(std::numbers::e, exp) / (std::sqrt(2 * std::numbers::pi * sigmaSquare));
		}

		double BinomialDistributionMass(const size_t trials, const size_t successes, const double p)
		{
			return comb(successes, trials) * std::pow(p, successes) * std::pow(1.0 - p, trials - successes);
		}

		double BinomialDistributionRange(const size_t trials, const size_t bottom, const size_t top, const double p)
		{
			double result = 0.0;
			for (size_t beg = bottom; beg <= top; ++beg)
				result += BinomialDistributionMass(trials, beg, p);

			return result;
		}

		// Single sample
		double NormalDistributionZValue(const double x, const double mean, const double sigma)
		{
			return (x - mean) / sigma;
		}

		// Multiple samples
		double NormalDistributionZValueSample(const double x, const double mean, const double sigma, const size_t sample)
		{
			return (x - mean) / (sigma / std::sqrt(sample));
		}

		double CalculateNormalDistributionZValue(double x)
		{
			auto func = [](double x) { return std::pow(std::numbers::e, -std::pow(x, 2) / 2); };
			return SimpsonsRule(-10000.0, x, func) / std::sqrt(2 * std::numbers::pi);
		}

		double HypergeometricalDist(size_t successes, size_t failures, size_t trials, size_t desired)
		{
			return static_cast<double>(comb(desired, successes) * comb(trials - desired, failures)) / comb(trials, successes + failures);
		}

		double NegativeBinomialDist(size_t failures, size_t successes, double p)
		{
			return static_cast<double>(comb(successes - 1, failures + successes - 1)) * std::pow(p, successes) * std::pow(1.0 - p, failures);
		}

		double GeometricDist(size_t failures, double p)
		{
			return std::pow(1.0 - p, failures) * p;
		}

		// mu average of population, x desired result
		double PoissonDist(double mu, unsigned long x)
		{
			return std::pow(std::numbers::e, -mu) * std::pow(mu, x) / fact(x);
		}

		double GetFisherDist(double m, double n)
		{
			std::fisher_f_distribution dist(m, n);
			return dist(engine);
		}

		double GetChiSquare(double m)
		{
			std::chi_squared_distribution<double> dist(m);
			return dist(engine);
		}

		double CalculateZScoreWithConfidenceLevel(size_t percentage)
		{
			double percentageDivided = (percentage) / 100.0;
			const auto alpha = (1.0 + percentageDivided) / 2.0;
			const auto negativeAlpha = 1.0 - alpha;
			double lowestValue = -3.7;
			const double pointInterval = 0.01;
			while (lowestValue <= 3.7)
			{
				const auto converted = CalculateNormalDistributionZValue(lowestValue);
				if (std::abs(converted - negativeAlpha) <= 0.0001) // does not work with epsilon
				{
					return lowestValue;
				}
				lowestValue += pointInterval;
			}
			return 0.0;
		}

		double tWithDegreesOfFreedom(double t, size_t df) 
		{
			if (t < 0) 
				return 0.5 * std::tgamma((df + 1) / 2.0) / (std::sqrt(df * std::numbers::pi) * std::tgamma(df / 2.0) * std::pow(1 + (t * t) / df, (df + 1) / 2.0));

			return 1.0 - 0.5 * std::tgamma((df + 1) / 2.0) / (std::sqrt(df * std::numbers::pi) * std::tgamma(df / 2.0) * std::pow(1 + (t * t) / df, (df + 1) / 2.0));
		}

		double FindTValue(size_t df, double alpha, bool two_tailed) 
		{
			if (two_tailed) 
			{
				alpha /= 2.0;
			}

			double lowerBound = -10.0; 
			double upperBound = 10.0;
			double tValue = 0.0;

			while (upperBound - lowerBound > 1e-6) // Do not use numeric_limits
			{
				tValue = (lowerBound + upperBound) / 2.0;
				if (tWithDegreesOfFreedom(tValue, df) < (1.0 - alpha))
				{
					lowerBound = tValue;  
				}
				else 
				{
					upperBound = tValue;  
				}
			}
			return tValue;
		}

		// Hypothesis test, if it is true, we should reject null hypothesis
		enum class TailTest {Left, Right, Both};
		void CheckAlternativeHypothesis(double x, double mean, double stddev, size_t sampleSize, TailTest test)
		{
			constexpr auto confidenceLevel = 95;
			const auto intervals = std::abs(CalculateZScoreWithConfidenceLevel(confidenceLevel)); // slow but it works 
			if (sampleSize > 30) // we do z test when we know stddev and when sampleSize > 30
			{
				const auto z = NormalDistributionZValueSample(x, mean, stddev, sampleSize);
				if (test == TailTest::Both)
				{
					if (std::abs(z) >= intervals)
					{
						std::cout << "Null hypothesis should be rejected" << std::endl;
						return;
					}
				}
				else if (test == TailTest::Left)
				{
					if (z < intervals)
					{
						std::cout << "Null hypothesis should be rejected" << std::endl;
						return;
					}
				}
				else if (test == TailTest::Right)
				{
					if (z > intervals)
					{
						std::cout << "Null hypthesis should be rejected" << std::endl;
						return;
					}
				}
				std::cout << "Null hypothesis is right!" << std::endl;
			}
			else
			{
				const auto t = NormalDistributionZValueSample(x, mean, stddev, sampleSize);
				const auto degreeOfFreedom = sampleSize - 1;
				const auto significanceLevel = (1.0 - confidenceLevel);
				const auto criticalT = FindTValue(degreeOfFreedom, significanceLevel, test == TailTest::Both);

				if (test == TailTest::Both)
				{
					if (std::abs(t) >= criticalT)
					{
						std::cout << "Null hypothesis should be rejected" << std::endl;
						return;
					}
				}
				else if (test == TailTest::Left)
				{
					if (t <= criticalT)
					{
						std::cout << "Null hypothesis should be rejected" << std::endl;
						return;
					}
				}
				else if (test == TailTest::Right)
				{
					if (t >= criticalT)
					{
						std::cout << "Null hypothesis should be rejected" << std::endl;
						return;
					}
				}

				std::cout << "Null hypothesis is right" << std::endl;
			}
		}
		
		double FicherFValue(double variance1, double variance2)
		{
			return variance1 / variance2;
		}
		// Tests for variance or standard deviation
		// f test and chi square test
		// F test for two variances from different population
		// Chi squared is when testing population variance against a 
		// specified method
		// Testing goodness of fit of some probability distribution

		double ChiSquaredTestStatistics(size_t n, double variance, double std)
		{
			return (n - 1) * variance / std;
		}

		double CriticalValueForChiSquared([[maybe_unused]]double confidence, [[maybe_unused]] double testStatistics, [[maybe_unused]] size_t degreesOfFreedom)
		{
			// look at the table.........
			return 0.0;
		}




		// Anova used for severals means 
		// Checking one or more samples has different mean than population

	private:
		std::random_device device{};
		std::mt19937 engine{ device()};
	};

	static Random random{};

	auto GetUniformIntNDistributions(size_t N, int min, int max)
	{
		std::vector<long long> results(N, 0u);
		for (size_t i{ 0u }; i < N; ++i)
			results[i] = random.GetUniformInt(min, max);

		return results;
	}

};


#pragma region Regular

template <typename VecType>
void PrintVec(const VecType& vec)
{
	for (const auto& elem : vec)
		std::cout << elem << ", ";

	std::cout << std::endl;
}


void print_break(const std::vector<size_t>& widths)
{
	const std::size_t margin = 1;
	std::cout.put('+').fill('-');
	for (std::size_t w : widths)
	{
		std::cout.width(w + margin * 2);
		std::cout << '-' << '+';
	}
	std::cout.put('\n').fill(' ');
};

std::vector<size_t> calculate_column_widths(const std::vector<std::vector<std::string>>& table)
{
	std::vector<size_t> widths;
	widths.resize(table.size() + 1);

	for (const auto& row : table)
		for (size_t i = 0; i != row.size(); ++i)
			widths[i] = 10;
	return widths;
}

template<typename T>
void print_row(const std::vector<T>& row, const std::vector<size_t>& widths)
{
	std::cout << '|';
	for (size_t i = 0; i != row.size(); ++i)
	{
		std::cout << ' ';
		std::cout.width(widths[i]);
		std::cout << row[i] << " |";
	}
	std::cout << '\n';
};

//Centralna granicna teorema
//Za bilo koju populaciju, proseci nasumicnih uzoraka ce imati normalnu raspodelu

void Simulation1(size_t examples)
{
	constexpr auto tests = 100u;
	std::vector<float> averages(tests, 0.0f);
	for (size_t test{ 0u }; test < tests; ++test)
	{
		const auto sims = statistics::GetUniformIntNDistributions(examples, 1, 6); // dice
		averages[test] = statistics::avg(sims);
	}

	const auto bins = statistics::CreateBins(averages);
	statistics::DrawBonsHistogram(averages, bins);
}

void Simulation2(size_t sample)
{
	std::vector<long long> x, y;
	for (size_t i{ 0u }; i < sample; ++i)
	{
		auto xRand = statistics::random.GetUniformFloat(-24, 24);
		auto yRand = statistics::random.GetUniformFloat(-24, 24);
		auto result = static_cast<size_t>(std::sqrt(std::pow(xRand, 2) + std::pow(yRand, 2)));
		if (result <= 24)
		{
			x.push_back(static_cast<long long>(xRand));
			y.push_back(static_cast<long long>(yRand));
		}
	}

	statistics::DrawCoordinate(x, y);

}

template<typename Type>
MATRIX<Type>::MATRIX(size_t rows, size_t columns, const std::string& rowNames, const std::string& columnNames)
{
	matrix.resize(rows);
	for (size_t i{ 0u }; i < rows; ++i)
		matrix[i].resize(columns);

	for (size_t i = 0; i < rows; ++i)
	{
		for (size_t j = 0; j < columns; ++j)
		{
			matrix[i][j] = static_cast<Type>(0);
		}
	}

	this->columnName = columnNames;
	this->rowName = rowNames;
}

template<typename Type>
MATRIX<Type>::MATRIX(const MATRIX& matrix)
{
	this->matrix.resize(matrix.GetRows());
	for (size_t i{ 0u }; i < matrix.GetRows(); ++i)
		this->matrix[i].resize(matrix.GetColumns());

	for (uint32_t i = 0; i < matrix.GetRows(); ++i)
	{
		for (uint32_t j = 0; j < matrix.GetColumns(); ++j)
		{
			this->matrix[i][j] = matrix[i][j];
		}
	}
}

template<typename Type>
MATRIX<Type>::MATRIX(const std::vector<typename MATRIX<Type>::VECTOR>& vectors)
{
	this->matrix = vectors;
}

template <typename Type>
template <typename OtherType>
MATRIX<Type>::MATRIX(const MATRIX<typename OtherType>& other)
{
	matrix.clear();
	matrix.resize(other.GetRows());
	for (auto& row : matrix)
		row.resize(other.GetColumns());

	for (uint32_t i{}; i < other.GetRows(); ++i)
	{
		for (uint32_t j{}; j < other.GetColumns(); ++j)
		{
			matrix[i][j] = static_cast<Type>(other[i][j]);
		}
	}
}

template<typename Type>
const MATRIX<Type>::VECTOR& MATRIX<Type>::operator[](uint32_t index) const
{
	return matrix.at(index);
}

template<typename Type>
typename MATRIX<Type>::VECTOR& MATRIX<Type>::operator[](uint32_t index)
{
	return matrix[index];
}

template<typename Type>
bool MATRIX<Type>::operator==(const MATRIX& other) const
{
	for (size_t i = 0; i < other.GetRows(); ++i)
	{
		for (size_t j = 0; j < other.GetColumns(); ++j)
		{
			if (matrix[i][j] != other[i][j])
				return false;
		}
	}

	return true;
}

template<typename Type>
MATRIX<Type> MATRIX<Type>::operator-() const
{
	MATRIX<Type> newMatrix(*this);

	for (size_t i{ 0u }; i < matrix.size(); ++i)
		for (size_t j{ 0u }; j < matrix[i].size(); ++i)
			newMatrix[i][j] = -matrix[i][j];

	return newMatrix;
}

template<typename Type>
MATRIX<Type> MATRIX<Type>::GetNthDegree(size_t degree)
{
	if (degree == 1 || degree == 0)
		return *this;

	const auto newMatrix = GetNthDegree(degree / 2);
	if (degree % 2 == 0)
	{
		return newMatrix * newMatrix;
	}
	else
	{
		return *this * newMatrix * newMatrix;
	}

}

template <typename Type>
MATRIX<Type> operator+(const MATRIX<Type>& left, const MATRIX<Type>& right)
{
	MATRIX<Type> newMatrix(left.matrix.size(), left.matrix[0].size());

	for (size_t i{ 0u }; i < newMatrix.size(); ++i)
		for (size_t j{ 0u }; j < newMatrix[i].size(); ++i)
			newMatrix[i][j] = left[i][j] + right[i][j];

	return newMatrix;
}

template <typename Type>
MATRIX<Type> operator-(const MATRIX<Type>& left, const MATRIX<Type>& right)
{
	MATRIX<Type> newMatrix(left.matrix.size(), left.matrix[0].size());

	for (size_t i{ 0u }; i < newMatrix.size(); ++i)
		for (size_t j{ 0u }; j < newMatrix[i].size(); ++i)
			newMatrix[i][j] = left[i][j] - right[i][j];

	return newMatrix;
}

template <typename Type>
MATRIX<Type> operator*(const MATRIX<Type>& left, const MATRIX<Type>& right)
{
	if (left.GetColumns() != right.GetRows())
		throw std::runtime_error("this.columns should be like other.rows! \n");

	MATRIX<Type> result(left.GetRows(), right.GetColumns());
	for (uint32_t i{}; i < result.GetRows(); ++i)
	{
		for (uint32_t j{}; j < result.GetColumns(); ++j)
		{
			const auto& multRow = left.GetRow(i);
			const auto column = right.GetColumn(j);

			for (size_t k{}; k < std::size(multRow); ++k)
			{
				result[i][j] += multRow[k] * column[k];
			}
		}
	}
	return result;
}


template <typename Type>
MATRIX<Type> operator*(double left, const MATRIX<Type>& right)
{
	MATRIX<Type> newMatrix(left.matrix.size(), left.matrix[0].size());

	for (size_t i{ 0u }; i < newMatrix.size(); ++i)
		for (size_t j{ 0u }; j < newMatrix[i].size(); ++i)
			newMatrix[i][j] = left * right[i][j];

	return newMatrix;
}

template <typename Type>
MATRIX<Type> operator*(const MATRIX<Type>& left, double right)
{
	MATRIX<Type> newMatrix(left.matrix.size(), left.matrix[0].size());

	for (size_t i{ 0u }; i < newMatrix.size(); ++i)
		for (size_t j{ 0u }; j < newMatrix[i].size(); ++i)
			newMatrix[i][j] = left[i][j] * right;

	return newMatrix;
}

template <typename Type>
MATRIX<Type> operator/(const MATRIX<Type>& left, double right)
{
	MATRIX<Type> newMatrix(left.matrix.size(), left.matrix[0].size());

	for (size_t i{ 0u }; i < newMatrix.size(); ++i)
		for (size_t j{ 0u }; j < newMatrix[i].size(); ++i)
			newMatrix[i][j] = left[i][j] / right;

	return newMatrix;
}

template<typename Type>
MATRIX<Type>& MATRIX<Type>::operator+=(const MATRIX<Type>& other)
{
	for (size_t i{ 0u }; i < other.GetRows(); ++i)
		for (size_t j{ 0u }; j < other.GetColumns(); ++i)
			matrix[i][j] += other[i][j];

	return *this;
}

template<typename Type>
MATRIX<Type>& MATRIX<Type>::operator-=(const MATRIX<Type>& other)
{
	for (size_t i{ 0u }; i < other.GetRows(); ++i)
		for (size_t j{ 0u }; j < other.GetColumns(); ++i)
			matrix[i][j] -= other[i][j];

	return *this;
}

template<typename Type>
MATRIX<Type>& MATRIX<Type>::operator*=(const MATRIX<Type>& other)
{
	for (size_t i{ 0u }; i < other.GetRows(); ++i)
		for (size_t j{ 0u }; j < other.GetColumns(); ++i)
			matrix[i][j] *= other[i][j];

	return *this;
}

template<typename Type>
MATRIX<Type>& MATRIX<Type>::operator*=(double other)
{
	for (size_t i{ 0u }; i < other.GetRows(); ++i)
		for (size_t j{ 0u }; j < other.GetColumns(); ++i)
			matrix[i][j] *= other;

	return *this;
}

template<typename Type>
MATRIX<Type>& MATRIX<Type>::operator/=(double other)
{
	for (size_t i{ 0u }; i < other.GetRows(); ++i)
		for (size_t j{ 0u }; j < other.GetColumns(); ++i)
			matrix[i][j] /= other;

	return *this;
}

template<typename Type>
MATRIX<Type>& MATRIX<Type>::operator-=(double other)
{
	for (size_t i{ 0u }; i < other.GetRows(); ++i)
		for (size_t j{ 0u }; j < other.GetColumns(); ++i)
			matrix[i][j] -= other;

	return *this;
}

template<typename Type>
MATRIX<Type>& MATRIX<Type>::operator+=(double other)
{
	for (size_t i{ 0u }; i < other.GetRows(); ++i)
		for (size_t j{ 0u }; j < other.GetColumns(); ++i)
			matrix[i][j] += other;

	return *this;
}

template<typename Type>
typename MATRIX<Type>::VECTOR MATRIX<Type>::GetColumn(size_t index) const
{
	if (index >= matrix[0].size())
		throw std::runtime_error("index out of bounds");

	std::vector<Type> result;
	for (size_t i{ 0u }; i < matrix.size(); ++i)
	{
		result.push_back(matrix[i][index]);
	}

	return result;
}

template<typename Type>
typename MATRIX<Type>::VECTOR MATRIX<Type>::GetRow(size_t index) const
{
	if (index >= matrix.size())
		throw std::runtime_error("index out of bounds");

	return matrix[index];
}

template<typename Type>
double MATRIX<Type>::GetDeterminant() const
{
	if (GetRows() != GetColumns())
		throw std::runtime_error("Matrix should be squared");

	double determinant = 0.0;

	if (GetRows() <= 6)
	{
		// Do laibniz for small matrices. For bigger matrices use LU decomposition or Gauss 
		determinant = CalculateAndDivideMatrices(this->matrix);
	}
	else
	{

	}
	
	return determinant;
}

template<typename Type>
void MATRIX<Type>::PrintMatrix() const
{
	if (matrix.size() > 0 && matrix[0].size() > 0)
	{
		std::vector<std::vector<std::string>> table;
		table.resize(matrix.size());
		for (size_t i = 0; i < table.size(); ++i)
			table[i].resize(matrix[i].size() + 1);

		for (size_t i = 0; i < GetRows(); i++)
		{
			for (size_t j = 1; j < GetColumns() + 1; j++)
			{
				table[i][j] = std::to_string(matrix[i][j-1]);
			}
		}

		for (size_t i = 0; i < GetRows(); ++i)
		{
			table[i][0] = rowName + std::to_string(i);
		}

		auto widths = calculate_column_widths(table);
		std::cout.setf(std::ios::left, std::ios::adjustfield);
		print_break(widths);
		
		std::vector<std::string> columns;
		columns.push_back(" ");
		for (size_t i = 0; i < matrix[0].size(); ++i)
		{
			columns.emplace_back(columnName + " " + std::to_string(i));
		}

		for (size_t i = 0; i < columns.size(); ++i)
		{
			widths[i] = std::max(widths[i], columns[i].size());
		}

		print_row(columns, widths);
		print_break(widths);
		for (const auto& row : table)
			print_row(row, widths);
		print_break(widths);

	}
}

template<typename Type>
MATRIX<Type> MATRIX<Type>::GetTransposed() const
{
	MATRIX<Type> matrix(GetColumns(), GetRows());
	for (uint32_t i{}; i < GetColumns(); ++i)
	{
		for (uint32_t j{}; j < GetRows(); ++j)
		{
			matrix[i][j] = this->matrix[j][i];
		}
	}

	return matrix;
}


template<typename Type>
double MATRIX<Type>::SmallMatrixLaibnizDeterminant(const MATRIX<Type>& type)
{
	if (type.GetRows() == type.GetColumns() && type.GetRows() == 2)
	{
		return (type[0][0] * type[1][1] - type[0][1] * type[1][0]);
	}

	return 0.0;
}

template<typename Type>
double MATRIX<Type>::CalculateAndDivideMatrices(const MATRIX& type)
{
	if (type.GetRows() == 2)
		return SmallMatrixLaibnizDeterminant(type);

	int sign = 1;
	double result = 0.0;
	for (uint32_t j{}; j < type.GetColumns(); j++)
	{
		result += sign * (type[0][j] * CalculateAndDivideMatrices(type.CreateMatrixWithoutColumn(j)));
		sign *= -1;
	}

	return result;
}

template<typename Type>
MATRIX<Type> MATRIX<Type>::CreateMatrixWithoutColumn(size_t column) const
{
	MATRIX<Type> result(GetRows() - 1, GetColumns() - 1);
	size_t resultPtrI{}, resultPtrJ{};
	for (uint32_t i{}; i < GetRows(); ++i)
	{
		if (i == 0)
			continue;

		resultPtrJ = 0u;
		for (uint32_t j{}; j < GetColumns(); ++j)
		{
			if (j == column)
				continue;

			result[resultPtrI][resultPtrJ++] = this->matrix[i][j];
		}
		resultPtrI++;
	}

	return result;
}



#pragma endregion
using namespace statistics;
auto main() -> int
{
#pragma region COMMENTED
	/*std::vector<int> a{ 12,3,4,5,6,11,12 };
	std::vector < std::string> c{ "a", "b", "c", "d", "e", "f", "g" };
	std::vector<float> b{
		21.0f,
		21.0f,
		22.8f,
		21.4f,
		18.7f,
		18.1f,
		14.3f,
		24.4f,
		22.8f,
		19.2f,
		17.8f,
		16.4f,
		17.3f,
		15.2f,
		10.4f,
		10.4f,
		14.7f,
		32.4f,
		30.4f,
		33.9f,
		21.5f,
		15.5f,
		15.2f,
		13.3f,
		19.2f,
		27.3f,
		26.0f,
		30.4f,
		15.8f,
		19.7f,
		15.0f,
		21.4f
	};
	std::vector<std::string> d;
	for (const auto& vv : b)
	{
		d.push_back(std::to_string(vv));
	}

	std::vector<float> borders{ 20.0f, 22.0f, 30.0f };
	std::array machine1{ 150, 151, 152, 152, 151, 150 };
	std::array machine2{ 153, 152, 148, 151, 149, 152 };
	std::array machine3{ 156, 154, 155, 156, 157, 155 };*/

	//std::cout << statistics::sum(a, [](int a) {return a % 2 == 0; });
	//std::cout << statistics::avg(a, [](int a) {return a % 2 == 0; });
	//std::cout << statistics::mean(a);
	//std::cout << statistics::dev(a);
	//std::cout << statistics::var(a);
	//std::cout << statistics::range(a);
	//std::cout << statistics::min(a);
	//std::cout << statistics::stderror(a);
	//std::cout << statistics::skew1(b);
	//std::cout << statistics::skew2(b);
	//std::cout << statistics::kurtosis1(b);
	//std::cout << statistics::kurtosis2(b);
	//statistics::DrawHistogram(b, d);
	//statistics::DrawBonsHistogram(b, borders);
	//statistics::DrawCoordinate(std::vector{0, 1, 2, 3, -4}, std::vector{0, 1, 2, 3, -4});
	//statistics::PrintVec(statistics::CreateBins(b));
	//std::cout << statistics::fact(0);
	//std::cout << statistics::perm(4, 10, true);
	//std::cout << statistics::comb(2, 10);
	//auto randomDiceSimulation = statistics::GetUniformIntNDistributions(100, 1, 6);
	//statistics::PrintVec(randomDiceSimulation);

	//Simulation1(10);
	//Simulation1(100);
	//Simulation1(1000);
	//Simulation1(1);
	//Simulation2(10);
	//Simulation2(100);
	//Simulation2(10000);
	//std::cout << statistics::random.NormalDistributionDensity(0.5f, 1.0f, 0.0f);
	//std::cout << statistics::random.BinomialDistributionMass(10, 5, 0.5);
	//std::cout << statistics::random.BinomialDistributionRange(10, 0, 5, 0.5);
	//std::cout << statistics::TrapezoidalRule(0, 2 * std::numbers::pi, [](double x) {return std::sin(x) * std::sin(x) / (2 * std::numbers::pi); });
	//std::cout << std::endl;
	//std::cout << statistics::SimpsonsRule(0, 2 * std::numbers::pi, [](double x) {return std::sin(x) * std::sin(x) / (2 * std::numbers::pi); });
	//std::cout << statistics::random.CalculateNormalDistributionZValue(1.0);
	//std::cout << statistics::random.CalculateNormalDistributionZValue(0.0);
	//std::cout << statistics::random.HypergeometricalDist(5, 5, 3, 2);
	//std::cout << statistics::random.NegativeBinomialDist(3, 2, 0.5);
	//std::cout << statistics::random.GeometricDist(4, 0.5);
	//std::cout << statistics::random.PoissonDist(3.6, 7); 
	//std::cout << statistics::random.NormalDistributionZValue(102.5, 100, 5);
	//std::cout << statistics::random.CalculateNormalDistributionZValue(-1.96);
	/*std::cout << */
	//statistics::random.CheckAlternativeHypothesis(78, 80, 2.5, 40);
	//std::cout << statistics::random.CalculateZScoreWithConfidenceLevel(97);
	//std::cout << statistics::random.FindTValue(10, 0.05, true);
	//std::cout << statistics::CalculateSumOfSquaresWithin(std::vector{ machine1 ,machine2 ,machine3 }) << std::endl; 
	//std::cout << statistics::CalculateSumOfSquaresBetween(std::vector{ machine1 ,machine2 ,machine3 }) << std::endl;
	//std::cout << statistics::MeanSSBetween(std::vector{ machine1 ,machine2 ,machine3 });
	//std::cout << statistics::CalculateFRationValue(std::vector{ machine1 ,machine2 ,machine3 });
	//statistics::ANOVA(std::vector{ machine1 ,machine2 ,machine3 });
	//std::cout << statistics::getCriticalChiSquared(0.05, 4);
	//statistics::GoodnessOfFitTest(std::vector{51, 52, 49, 83, 48}, std::vector{ 50, 50, 50, 50, 50 });
	//   MATRIX m(5, 4, "Students", "Grades");
	//size_t br = 0;
	//for (size_t i = 0; i < 5; i++)
	//{
	//	for (size_t j = 0; j < 4; j++)
	//	{
	//		m[i][j] = ++br;
	//	}
	//}
	//m.PrintMatrix();

	//PrintVec(m.GetColumn(0));
	//PrintVec(m.GetRow(0));
	//statistics::ContigencyTable(m);
//MATRIX m(std::vector{ std::vector<size_t>{22,26,23}, std::vector<size_t>{28,62,26}, std::vector<size_t>{72,22,66} });
//m.SetRowName("Shift");
//m.SetColumnName("Operator");
//m.PrintMatrix();
//statistics::ContigencyTable(m);
//
//std::vector x{ 20, 24, 46, 62, 22, 37, 45, 27, 65, 23 };
//std::vector y{ 40, 55, 69, 83, 27, 44, 61, 33, 71, 37 };
//std::cout << GetPredictionOFY(x, y, 22);
//MATRIX<size_t> m1(5, 5);
//MATRIX<double> m2(m1);
//MATRIX<size_t> m1(std::vector{ std::vector<size_t>{2,3}, std::vector<size_t>{4,5}, std::vector<size_t>{6,7} });
//MATRIX<size_t> m2(std::vector{ std::vector<size_t>{2,4, 6}, std::vector<size_t>{3,5,7} });
//auto m3(m1* m2);
////m3.PrintMatrix();
//MATRIX<size_t> m2(std::vector{ std::vector<size_t>{2,2, 2}, std::vector<size_t>{2,2,2}, std::vector<size_t>{2,2,2} });
////m2.GetNthDegree(3).PrintMatrix();
//MATRIX<size_t> m2(std::vector{ std::vector<size_t>{3, 8}, std::vector<size_t>{4, 6} });
//m2.PrintMatrix();
//std::cout << MATRIX<int>::SmallMatrixLaibnizDeterminant(m2);
//MATRIX<int> m2(std::vector{ std::vector<int>{4,3,2, 2}, std::vector<int>{0,1,-3, 3}, std::vector<int>{0,-1,3, 3}, 
//	std::vector<int>{0,3,1,1} });
//auto s2 = m2.CreateMatrixWithoutColumn(1);
//m2.PrintMatrix();
//std::cout << m2.GetDeterminant();
#pragma endregion
MATRIX<int64_t> test(6, 6);
for (uint32_t i{}; i < test.GetColumns(); ++i)
{
	for (uint32_t j{}; j < test.GetRows(); ++j)
	{
		test[i][j] = statistics::random.GetUniformInt(-200, 200);
	}
}

test.PrintMatrix();
std::cout << test.GetDeterminant();
}
