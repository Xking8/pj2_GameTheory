#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss(args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			property[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string name() const {
		auto it = property.find("name");
		return it != property.end() ? std::string(it->second) : "unknown";
	}
protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> property;
};

/**
 * evil (environment agent)
 * add a new random tile on board, or do nothing if the board is full
 * 2-tile: 90%
 * 4-tile: 10%
 */
class rndenv : public agent {
public:
	rndenv(const std::string& args = "") : agent("name=rndenv " + args) {
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
	}

	virtual action take_action(const board& after) {
		int space[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
		std::shuffle(space, space + 16, engine);
		for (int pos : space) {
			if (after(pos) != 0) continue;
			std::uniform_int_distribution<int> popup(0, 9);
			int tile = popup(engine) ? 1 : 2;
			return action::place(tile, pos);
		}
		return action();
	}

private:
	std::default_random_engine engine;
};

/**
 * TODO: player (non-implement)
 * always return an illegal action
 */
class player : public agent {
public:
	player(const std::string& args = "") : agent("name=player " + args), alpha(0.0025f) {
		episode.reserve(32768);
		if (property.find("seed") != property.end())
			engine.seed(int(property["seed"]));
		if (property.find("alpha") != property.end())
			alpha = float(property["alpha"]);

		if (property.find("load") != property.end())
			load_weights(property["load"]);
		// TODO: initialize the n-tuple network
		else {
			weights.push_back(std::pow(24,4));//weights[0]
			weights.push_back(std::pow(24,4));//weights[1]
			weights.push_back(std::pow(24,4));//weights[2]
			weights.push_back(std::pow(24,4));//weights[3]
			weights.push_back(std::pow(24,4));//weights[4]
			weights.push_back(std::pow(24,4));//weights[5]
			weights.push_back(std::pow(24,4));//weights[6]
			weights.push_back(std::pow(24,4));//weights[7]
		}
	}
	~player() {
		if (property.find("save") != property.end())
			save_weights(property["save"]);
	}

	virtual void open_episode(const std::string& flag = "") {
		episode.clear();
		episode.reserve(32768);
	}

	void update(board& after, double TDerror) {	
		for(int i=0;i<4;i++) {
			int index = 0;
			for(int j=0;j<4;j++) {
				index += after[i][j]*std::pow(24,3-j);
				
			}
			weights[i][index] += alpha*TDerror;// /8;
			//after[0][0]*std::pow(24,3)
		}
		for(int i=0;i<4;i++) {
			int index = 0;
			for(int j=0;j<4;j++) {
				index += after[j][i]*std::pow(24,3-j);
				
			}
			weights[i+4][index] += alpha*TDerror;// /8;
			//after[0][0]*std::pow(24,3)
		}
	}

	virtual void close_episode(const std::string& flag = "") {
		//std::cout<<"testing close episode:\n"<<episode[0].before<<"\ntaking action"<<episode[0].move<<":\n"<<episode[0].after<<std::endl;
		// TODO: train the n-tuple network by TD(0)
		//std::cout<<"..."<<episode.size()<<std::endl;
		for(int i = episode.size()-1; i>=0; i--)
		{	
			if(i==episode.size()-1)
			{
				//std::cout<<"testing close episode:\n"<<episode[i].before<<"\ntaking action"<<episode[i].move<<":\n"<<episode[i].after<<std::endl;
				update(episode[i].after,0);		

			}
			else
			{
				double TDerror = episode[i].reward + approx(episode[i].after)-approx(episode[i-1].after);
				update(episode[i-1].after, TDerror);
			}

			
		}
			
		/*for(long i=0;i<pow(24,4);i++)
			std::cout << weights[0][i] <<" ";
		std::cout << "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeE"<<std::endl;*/
	}
	virtual action take_action(const board& before) {
		action best;
		//int opcode[] = { 0, 1, 2, 3 };
		int max = -1;
		int max_action = -1;
		for(int i=0;i<4;i++)
		{
			int value=evaluate(before,i);
			if(value > max)
			{
				max_action = i;
				max = value;
			}
		}
		//std::cout<<"max_evaluation:"<< max <<std::endl;
		best = action(max_action);
		
		state_pack.before = before;
		state_pack.move = action(max_action);
		board after = before;
		double reward = compute_afterstate(after,max_action);
		state_pack.after = after;
		state_pack.reward = reward;
		episode.push_back(state_pack);
		//std::cout<<"max_action: "<<max_action<<std::endl;
		
		// TODO: select a proper action
		// TODO: push the step into episode
		//best = action(0);
		return best;
	}
	double evaluate(const board& before, int act)
	{
		board after = before;
		double reward = compute_afterstate(after,act);
		state_pack.after = after;
		state_pack.reward = reward;
		return reward+approx(after);
	}
	double compute_afterstate(board& after, int act) {
		int reward = after.move(act);
		return reward;
	}
	double approx(board& after) {
		//...
		double appr = 0;
		for(int i=0;i<4;i++) {
			int index = 0;
			for(int j=0;j<4;j++) {
				index += after[i][j]*std::pow(24,3-j);
				
			}
			appr += weights[i][index];
			//after[0][0]*std::pow(24,3)
		}
		for(int i=0;i<4;i++) {
			int index = 0;
			for(int j=0;j<4;j++) {
				index += after[j][i]*std::pow(24,3-j);
				
			}
			appr += weights[i+4][index];
			//after[0][0]*std::pow(24,3)
		}
		return appr;
	}
	

public:
	virtual void load_weights(const std::string& path) {
		std::ifstream in;
		in.open(path.c_str(), std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		size_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		weights.resize(size);
		for (weight& w : weights)
			in >> w;
		in.close();
	}

	virtual void save_weights(const std::string& path) {
		std::ofstream out;
		out.open(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		size_t size = weights.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : weights)
			out << w;
		out.flush();
		out.close();
	}

private:
	std::vector<weight> weights;

	struct state {
		// TODO: select the necessary components of a state
		board before;
		board after;
		//board after_of_next_state;
		action move;
		int reward;
	} state_pack;

	std::vector<state> episode;
	float alpha;

private:
	std::default_random_engine engine;
};
