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
#include <random>

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
	auto dev(const VecType& vec, bool isPopulation = true)
	{
		const auto size = std::size(vec);
		if (size > 1)
		{
			const auto avg = statistics::avg(vec);
			auto result{ 0.0f };
			for (auto elem{ 0u }; elem < size; ++elem)
				result += static_cast<float>(std::pow(vec[elem] - avg, 2));

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

	template <typename VecType>
	void PrintVec(const VecType& vec)
	{
		for (const auto& elem : vec)
			std::cout << elem << ", ";

		std::cout << std::endl;
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

		double NormalDistributionZValue(const double x, const double mean, const double sigma)
		{
			return (x - mean) / sigma;
		}

		double CalculateNormalDistributionZValue(double x)
		{
			auto func = [](double x) { return std::pow(std::numbers::e, -std::pow(x, 2) / 2); };
			return SimpsonsRule(-10000.0, x, func) / std::sqrt(2 * std::numbers::pi);
		}

		double HypergeometricalDist(size_t successes, size_t failures, size_t trials, size_t desired)
		{
			
		}

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
			x.push_back(xRand);
			y.push_back(yRand);
		}
	}

	statistics::DrawCoordinate(x, y);

}


auto main() -> int
{
	std::vector<int> a{ 12,3,4,5,6,11,12 };
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
}