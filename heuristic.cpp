#pragma GCC optimize("O3")

#define DEBUG true
// #define DEBUG false

#if !DEBUG
    #define NDEBUG // assertを無視
#endif

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <climits>
#include <vector>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <bitset>
#include <cmath>
#include <algorithm>
#include <string>
#include <queue>
#include <stack>
#include <list>
#include <numeric>
#include <cassert> // assert();
#include <iomanip> // cout << setprecision(15); cout << setfill('0') << std::right << setw(3);
#include <fstream>
#include <random>
#include <chrono>
#include <thread>
#include <optional> // optional<int> f = nullopt; if(f) f.value();
#include <regex> // regex_replace("target", regex("old"), "new");
#include <filesystem>

#define MY_PI     3.14159265358979323846
#define MY_E      2.7182818284590452354
#define INF     (INT_MAX / 2)
#define LINF    (LLONG_MAX / 2)
#define FOR(i, a, b) for(int i = (a); i < (b); ++i)
#define RFOR(i, a, b) for(int i = (b) - 1; i >= (a); --i)
#define REP(i, n) FOR(i, 0, n)
#define RREP(i, n) RFOR(i, 0, n)
#define EACH(e, v) for(auto &e : v)
#define ITR(it, v) for(auto it = (v).begin(); it != (v).end(); ++it)
#define RITR(it, v) for(auto it = (v).rbegin(); it != (v).rend(); ++it)
#define CASE break; case
#define ALL(v)  (v).begin(), (v).end()
#define RALL(v) (v).rbegin(), (v).rend()
#define SZ(v) int((v).size())
#define EXIST(s, e) ((s).find(e) != (s).end())
#define SORT(v) sort((v).begin(), (v).end())
#define RSORT(v) sort((v).rbegin(), (v).rend())
#define SUM(v, type) accumulate((v).begin(), (v).end(), (type) 0)
#define MIN(v) (*min_element((v).begin(), (v).end()))
#define MAX(v) (*max_element((v).begin(), (v).end()))
#define ARGMIN(v) (min_element((v).begin(), (v).end()) - (v).begin())
#define ARGMAX(v) (max_element((v).begin(), (v).end()) - (v).begin())
#define REVERSE(v) reverse((v).begin(), (v).end())
#define ARANGE(v) iota((v).begin(), (v).end(), 0)
#define FILTER(src, tgt, func) copy_if(begin(src), end(src), back_inserter(tgt), func); // func = [](type x){return 条件;}
#define CTOI(c) (c - '0')
#define HEADSTR(str, n) str.substr(0, (n))
#define TAILSTR(str, n) str.substr((str).length() - (n))
#define CONTAINS(str, c) ((str).find(c) != string::npos)
#define INSPOS(v, a) (lower_bound((v).begin(), (v).end(), a) - (v).begin())
// デバッグ用
#define dump(x)  cerr << #x << " = " << (x) << endl;
#define debug(x) cerr << #x << " = " << (x) << " (L" << __LINE__ << ")" << " " << __FILE__ << endl;

using namespace std;

template<class T> vector<size_t> argsort(const vector<T> &vec, bool asc=true){
    vector<size_t> index(vec.size()); iota(index.begin(), index.end(), 0);
    sort(index.begin(), index.end(), [&vec, &asc](size_t i, size_t j){return asc ? (vec[i] < vec[j]):(vec[i] > vec[j]);});
    return index;
}
// 表示系
template<class T1, class T2> ostream& operator<<(ostream& os, const pair<T1, T2>& p) {
    os << "(" << p.first << ", " << p.second << ")";
    return os;
}
template<class... T> ostream& operator<<(ostream& os, const tuple<T...>& t) {
    os << "("; apply([&os](auto&&... args) {((os << args << ", "), ...);}, t);
    os << ")"; return os;
}
template<class T> ostream& operator<<(ostream& os, const vector<T>& vec) {
    os << "[ "; for ( const T& item : vec ) os << item << ", ";
    os << "]"; return os;
}
template<class T> ostream& operator<<(ostream& os, const set<T>& s) {
    os << "{ "; for ( const T& item : s ) os << item << ", ";
    os << "}"; return os;
}
template<class T> ostream& operator<<(ostream& os, const multiset<T>& s) {
    os << "{ "; for ( const T& item : s ) os << item << ", ";
    os << "}"; return os;
}
template<class T1, class T2> ostream& operator<<(ostream& os, const map<T1, T2>& m) {
    os << "{ "; for ( const auto &[key, value] : m ) os << key << ":"<< value << ", ";
    os << "}"; return os;
}
template <class Head> void OUT(Head&& head) {cout << head << endl;}
template <class Head, class... Tail> void OUT(Head&& head, Tail&&... tail) {cout << head << " ";OUT(forward<Tail>(tail)...);}
// 入力系
template<class T1, class T2> istream& operator>>(istream& is, pair<T1, T2>& p) {
    is >> p.first >> p.second;
    return is;
}
template<class... T> istream& operator>>(istream& is, tuple<T...>& t) {
    apply([&is](auto&&... args) {((is >> args), ...);}, t);
    return is;
}
template<class T> istream& operator>>(istream& is, vector<T>& vec) {
    for ( T& item : vec ) is >> item;
    return is;
}
// 集合演算
template<class T> set<T> operator&(const set<T>& a, const set<T>& b) {// 共通集合
    set<T> ans; set_intersection(a.begin(), a.end(), b.begin(), b.end(), inserter(ans, ans.end()));
    return ans;
}
template<class T> set<T> operator|(const set<T>& a, const set<T>& b) {// 和集合
    set<T> ans; set_union(a.begin(), a.end(), b.begin(), b.end(), inserter(ans, ans.end()));
    return ans;
}
template<class T> set<T> operator-(const set<T>& a, const set<T>& b) {// 差集合
    set<T> ans; set_difference(a.begin(), a.end(), b.begin(), b.end(), inserter(ans, ans.end()));
    return ans;
}

using LL = long long; using ULL = unsigned long long;
using VI = vector<int>; using VVI = vector<VI>; using VVVI = vector<VVI>;
using VL = vector<LL>; using VVL = vector<VL>; using VVVL = vector<VVL>;
using VB = vector<bool>; using VVB = vector<VB>; using VVVB = vector<VVB>;
using VD = vector<double>; using VVD = vector<VD>; using VVVD = vector<VVD>;
using VC = vector<char>; using VS = vector<string>;
using PII = pair<int,int>; using PLL = pair<LL,LL>; using PDD = pair<double,double>;
using MII = map<int,int>; using MLL = map<LL,LL>;
using SI = set<int>; using SL = set<LL>;
template<class T> using MAXPQ = priority_queue<T>;
template<class T> using MINPQ = priority_queue< T, vector<T>, greater<T> >;
// int -> str: to_string(i)
// str -> int: stoi(s)
// vec -> set: set<int> s(ALL(v));
// 1が立っている数: __builtin_popcount(i), __builtin_popcountll(i)
// 上位ビットの連続した0の数: __builtin_clz(i), __builtin_clzll(i) // i=0未定義
// 下位ビットの連続した0の数: __builtin_ctz(i), __builtin_ctzll(i) // i=0未定義

template <class Head> void IN(Head&& head) {cin >> head;}
template <class Head, class... Tail> void IN(Head&& head, Tail&&... tail) {cin >> head;IN(forward<Tail>(tail)...);}

// 乱数
// const int SEED = random_device()();
constexpr int SEED = 1;
mt19937 rand_engine(SEED);
// [0,1)
inline double get_rand(){
    uniform_real_distribution<double> rand01(0.0, 1.0);
    return rand01(rand_engine);
}
// [0, n)
inline int get_rand(int n){
    uniform_int_distribution<int> rand_n(0, n-1);
    return rand_n(rand_engine);
}
// [l, u)
inline int get_rand(int l, int u){
    uniform_int_distribution<int> rand_lu(l, u-1);
    return rand_lu(rand_engine);
}
// 累積和
template<class T>
inline vector<T> cumsum(const vector<T> &vec, bool zero_start){
    int n = SZ(vec) + zero_start;
    vector<T> cumsum(n);
    cumsum[0] = zero_start ? 0 : vec[0];
    FOR(i, 1, n)
        cumsum[i] = cumsum[i-1] + vec[i - zero_start];
    return cumsum;
}
template<class T>
inline int get_rand_index_with_cumsum_weight(const vector<T> &weight_cumsum){
    double p = get_rand() * weight_cumsum.back();
    return min((int)INSPOS(weight_cumsum, p), SZ(weight_cumsum)-1);
}
// 重みの割合でindex
template<class T>
inline int get_rand_index(const vector<T> &weight){
    auto weight_cumsum = cumsum(weight, false);
    return get_rand_index_with_cumsum_weight(weight_cumsum);
}
// [0,n)から重複なしでr個選ぶ
inline VI rand_choice(int n, int r){
    VI candidates(n);
    ARANGE(candidates);
    VI result;
    std::sample(candidates.begin(), candidates.end(), back_inserter(result), r, rand_engine);
    return result;
}

// 画面クリア
inline void clear_screen(){
    cout << "\x1b[0;0H";
}
// スリープ
inline void sleep(int msec){
    this_thread::sleep_for(chrono::milliseconds(msec));
}
// 数字ゼロ埋め文字列
inline string to_string_zerofill(int num, int digit){
    std::ostringstream sout;
    sout << std::setfill('0') << std::setw(digit) << num;
    return sout.str();
}

// CPU時間
class ClockTimer{
private:
    clock_t start_at, end_at;
public:
    ClockTimer(){}
    inline void start() { start_at = clock(); }
    inline void end() { end_at = clock(); }
    inline clock_t time() const { return 1000*(end_at-start_at) / CLOCKS_PER_SEC; }// ミリ秒
};

// chronoは処理系によってCPU時間か実時間か変わるため注意
class ChronoTimer{
private:
    chrono::high_resolution_clock::time_point start_at, end_at;
public:
    ChronoTimer(){}
    inline void start() { start_at = chrono::high_resolution_clock::now(); }
    inline void end() { end_at = chrono::high_resolution_clock::now(); }
    inline int64_t time() const { return chrono::duration_cast<chrono::milliseconds>(end_at - start_at).count(); }// ミリ秒
};

// 集合のハッシュ化: T=uint64_t, uint32_t, ULL, unsigned intなど
// → 系列版へ拡張
template<class T>
class ZobristHashing {
    vector<T> h;
    int size, nunique;
public:
    ZobristHashing(int nunique, int size, mt19937& rand_engine):size(size), nunique(nunique) {
        uniform_int_distribution<T> uniform(0, numeric_limits<T>::max());
        h.reserve(nunique*size);
        REP(i, nunique*size)
            h.emplace_back(uniform(rand_engine));
    }
    T hash(const VI &array) {
        T value = 0;
        REP(i, size){
            // (i, e) → i*nunique + e
            auto &e=array[i];
            value ^= h[i*nunique + e];
        }
        return value;
    }
};
vector<string> split_str(const string &str, char delim, bool ignore_empty){
    stringstream ss{str};
    string buf;
    vector<string> result;
    while (std::getline(ss, buf, delim)) {
        if (!ignore_empty || !buf.empty())
            result.emplace_back(buf);
    }
    if (!ignore_empty && !str.empty() && str.back()==delim)
        result.emplace_back("");
    return result;
}
bool file_exists(const std::string& name) {
    return std::filesystem::is_regular_file(name);
}


// データ
string puzzle_type;
int state_length;
map<string, int> label_mapping;
VI initial_state;
VI solution_state;
int allowed_action_num;
VS allowed_moves_name;
VVI allowed_moves;
int num_wildcards;
// 逆操作
VVI allowed_moves_inverse;

#define get_action(id) ((id)<allowed_action_num ? allowed_moves[id] : allowed_moves_inverse[(id)-allowed_action_num])
#define get_action_name(id) ((id)<allowed_action_num ? allowed_moves_name[id] : ("-"+allowed_moves_name[(id)-allowed_action_num]))

VI do_action(const VI& state, int action_id){
    auto s=state;
    auto &action = get_action(action_id);
    REP(i, state_length)
        s[i]=state[action[i]];
    return s;
}

VI simulation(const VI& state, const VI &actions){
    auto s=state;
    for(int a: actions)
        s = do_action(s, a);
    return s;
}

VI inverse(const VI &move){
    auto inv=move;
    REP(i, SZ(inv))
        inv[move[i]]=i;
    return inv;
}
// 
VI to_group_id;//(allowed_action_num);
VI to_order_in_group;//(allowed_action_num);

void devide_independent_action(){
    vector<SI> changes(allowed_action_num);
    REP(i, allowed_action_num){
        const auto& move=allowed_moves[i];
        REP(j, SZ(move))if(j!=move[j]){
            changes[i].emplace(j);
            changes[i].emplace(move[j]);
        }
    }
    to_group_id.assign(allowed_action_num, 0);
    to_order_in_group.assign(allowed_action_num, 0);
    VVI group;
    vector<SI> group_set;
    REP(i, allowed_action_num){
        bool change=false;
        REP(j, SZ(group_set))if((changes[i]&group_set[j]).empty()){
            to_group_id[i]=j;
            to_order_in_group[i]=SZ(group_set[j]);
            group[j].emplace_back(i);
            group_set[j]=group_set[j]|changes[i];
            change=true;
            break;
        }
        if(!change){
            to_group_id[i]=SZ(group_set);
            to_order_in_group[i]=0;
            group.emplace_back(VI{i});
            group_set.emplace_back(changes[i]);
        }
    }
    dump(group)
}
VI to_rotate_num;
void inv_operation(){
    to_rotate_num.assign(allowed_action_num, 0);
    REP(a, allowed_action_num){
        VI index_arange(SZ(initial_state));
        ARANGE(index_arange);

        VI index(SZ(initial_state));
        ARANGE(index);
        int rotate;
        for(rotate=1;rotate<10000;++rotate){
            index=do_action(index, a);
            if(index_arange==index)
                break;
        }
        to_rotate_num[a]= rotate==10000 ? INF : rotate;
    }
    dump(to_rotate_num)
}

void data_load(istream &is){
    // データ読み込み ----------
    is  >> puzzle_type
        >> state_length;
    initial_state.assign(state_length, 0);
    solution_state.assign(state_length, 0);
    label_mapping.clear();
    REP(i, state_length){
        string label; is >> label;
        if(!label_mapping.contains(label))
            label_mapping[label]=SZ(label_mapping);
        initial_state[i]=label_mapping[label];
    }
    REP(i, state_length){
        string label; is >> label;
        assert(label_mapping.contains(label));
        solution_state[i]=label_mapping[label];
    }
    is >> allowed_action_num;
    allowed_moves_name.assign(allowed_action_num, "");
    allowed_moves.assign(allowed_action_num, VI(state_length));
    REP(i, allowed_action_num){
        is  >> allowed_moves_name[i]
            >> allowed_moves[i];
    }
    is >> num_wildcards;
    // 逆操作
    allowed_moves_inverse.assign(allowed_action_num, VI());
    REP(i, allowed_action_num){
        allowed_moves_inverse[i]=inverse(allowed_moves[i]);
    }
    // グループ分け
    devide_independent_action();
    // 
    inv_operation();
}


// 不一致数
int get_mistakes(const VI& state){
    int cnt=0;
    REP(i, SZ(state))
        cnt += state[i]!=solution_state[i];
    return cnt;
}
int get_mistakes(const VI& state,const VI& goal_state){
    int cnt=0;
    REP(i, SZ(state))
        cnt += state[i]!=goal_state[i];
    return cnt;
}

string action_decode(const VI& actions){
    string ans="";
    REP(i, SZ(actions)){
        int a=actions[i];
        ans += get_action_name(a);
        if(i+1!=SZ(actions))
            ans += ".";
    }
    return ans;
}


VI load_actions(const string &filename){
    ifstream ifs(filename);
    string str;
    ifs>>str;

    map<string, int> action_mapping;
    REP(i, allowed_action_num){
        action_mapping[allowed_moves_name[i]]=i;
        action_mapping["-"+allowed_moves_name[i]]=i+allowed_action_num;
    }

    VI actions;
    for(auto& a: split_str(str, '.', true))
        actions.emplace_back(action_mapping[a]);
    return actions;
}
void save_actions(const string &filename, const VI& actions){
    ofstream ofs(filename);
    auto ans = action_decode(actions);
    OUT(ans);
    ofs << ans;
    ofs.close();
}

// VI construct_actions(uint64_t init_hash, uint64_t last_hash, const unordered_map<uint64_t, tuple<VI,uint64_t,int>> &pushed){
VI construct_actions(uint64_t init_hash, uint64_t last_hash, const unordered_map<uint64_t, tuple<int,uint64_t,int>> &pushed){
    VI actions;
    auto h=last_hash;
    while(h!=init_hash){
        const auto &[_,next,a]=pushed.at(h);
        // const auto &[next,a]=pushed.at(h);
        actions.emplace_back(a);
        h=next;
    }
    REVERSE(actions);
    return actions;
}
optional<VI> search(const VI &start_state, const VI& goal_state, int current_best_size, bool strong_search, 
    int wildcards=0,
    size_t give_up=100000000){
    // state -> hash
    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);
    // hash -> {length, prev_hash, action_id}
    unordered_map<uint64_t, tuple<int, uint64_t,int>> pushed;
    // mistake, length, hash
    MINPQ<tuple<int, int, uint64_t>> pq;
    auto init_hash = zhash.hash(start_state);
    pushed[init_hash]={0, 0, -1};
    pq.emplace(get_mistakes(start_state, goal_state), 0, init_hash);
    auto goal_hash = zhash.hash(goal_state);
    int searched=0;
    while(!pq.empty()){
        auto [mistake, length, hash] = pq.top(); pq.pop();
        searched++;
        if(searched%100000==0)
            dump(searched)
        if(goal_hash==hash){
            auto actions=construct_actions(init_hash, hash, pushed);
            assert(simulation(start_state, actions)==goal_state);
            // OUT("solved");
            if(SZ(actions)<current_best_size)
                return actions;
            return nullopt;
        }
        if(wildcards!=0 && mistake <= wildcards){
            // OUT("solved using wildcards");
            auto actions=construct_actions(init_hash, hash, pushed);
            if(SZ(actions)<current_best_size)
                return actions;
            return nullopt;
        }
        if(strong_search && get<0>(pushed[hash])<length) continue;
        auto actions=construct_actions(init_hash, hash, pushed);
        auto state=simulation(start_state, actions);
        if(length<current_best_size)
            REP(a, allowed_action_num*2){
                auto next = do_action(state, a);
                auto next_hash = zhash.hash(next);
                if(strong_search){
                    if(pushed.contains(next_hash) && get<0>(pushed[next_hash])<=length)
                        continue;
                }else{
                    if(pushed.contains(next_hash))
                        continue;
                }
                pushed[next_hash]={length+1, hash, a};
                pq.emplace(get_mistakes(next, goal_state), length+1, next_hash);
                if(pq.size()%5000000==0)
                    dump(pq.size())
                if(pushed.size()%5000000==0)
                    dump(pushed.size())
                // 打ち切り
                if(pq.size()>give_up||pushed.size()>give_up)
                    return nullopt;
            }
    }
    return nullopt;
}

// 
int search(const string &output, bool strong_search, double improve_rate=1){
    // state -> hash
    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);
    // hash -> {length, prev_hash, action_id}
    unordered_map<uint64_t, tuple<int, uint64_t,int>> pushed;
    // mistake, length, hash
    MINPQ<tuple<int, int, uint64_t>> pq;
    // 保存してあるベスト解
    int current_best_size=INF;
    if(file_exists(output)){
        current_best_size=SZ(load_actions(output));
        dump(current_best_size)
    }
    auto init_hash = zhash.hash(initial_state);
    pushed[init_hash]={0, 0, -1};
    pq.emplace(get_mistakes(initial_state), 0, init_hash);
    auto goal_hash = zhash.hash(solution_state);
    int searched=0;
    while(!pq.empty()){
        auto [mistake, length, hash] = pq.top(); pq.pop();
        searched++;
        if(searched%100000==0)
            dump(searched)
        if(goal_hash==hash){
            auto actions=construct_actions(init_hash, hash, pushed);
            assert(simulation(initial_state, actions)==solution_state);
            OUT("solved");
            if(SZ(actions)<current_best_size){
                OUT("saved", current_best_size, "->", SZ(actions));
                save_actions(output, actions);
            }
            return length;
        }
        if(num_wildcards!=0 && mistake <= num_wildcards){
            OUT("solved using wildcards");
            auto actions=construct_actions(init_hash, hash, pushed);
            if(SZ(actions)<current_best_size){
                OUT("saved", current_best_size, "->", SZ(actions));
                save_actions(output, actions);
            }
            return length;
        }
        if(get<0>(pushed[hash])<length) continue;
        auto actions=construct_actions(init_hash, hash, pushed);
        auto state=simulation(initial_state, actions);
        if(length<current_best_size*improve_rate){
            REP(a, allowed_action_num*2){
                auto next = do_action(state, a);
                auto next_hash = zhash.hash(next);
                if(strong_search){
                    if(pushed.contains(next_hash) && get<0>(pushed[next_hash])<=length)
                        continue;
                }else{
                    if(pushed.contains(next_hash))
                        continue;
                }
                pushed[next_hash]={length+1, hash, a};
                pq.emplace(get_mistakes(next), length+1, next_hash);
                if(pq.size()%5000000==0)
                    dump(pq.size())
                if(pushed.size()%5000000==0)
                    dump(pushed.size())
                // // 打ち切り
                // if(pq.size()>100000000||pushed.size()>100000000)
                //     return INF;
            }
        }
    }
    return INF;
}

constexpr int FOUND=-1;
struct IDAstar{
    ZobristHashing<uint64_t> zhash;
    int bound, current_best;
    unordered_set<uint64_t> visit;
    // uint64_t goal_hash;
    VI goal_state;
    int wildcards;
    VI actions;
    int searched;
    IDAstar():zhash(SZ(label_mapping), state_length, rand_engine){
    }
    int h(const VI& state)const{
        return max(0, get_mistakes(state, goal_state)-wildcards);
    }
    optional<VI> ida_star(const VI &init_state, const VI &goal_state, int current_best, int wildcards){
        this->wildcards=wildcards;
        this->goal_state=goal_state;
        this->current_best=current_best;
        this->bound = h(init_state);
        // this->goal_hash=zhash.hash(goal_state);
        uint64_t init_hash=zhash.hash(init_state);
        visit.clear();
        visit.emplace(init_hash);
        VI state=init_state;
        VI path;
        while(1){
            dump(bound)
            searched=0;
            int t = search(state, path, 0, 0);
            if (t == FOUND) return actions;
            if (t == INF) return nullopt;
            bound=t;
            // bound++;
            dump(searched)
        }
    }
    int search(VI &state, VI &path, int g, int same_action_num){
        ++searched;
        int f = g + h(state);
        if (f > bound) return f;
        if (f==g){
            actions=path;
            return FOUND;
        }
        int minval=INF;
        if(g >= current_best) return minval;
        REP(a, allowed_action_num*2){
            int next_same_action_num=1;
            if(!path.empty()){
                // 逆操作
                if(a+allowed_action_num==path.back() || a==path.back()+allowed_action_num)
                    continue;
                // グループ順序
                int prev=(path.back()<allowed_action_num) ? path.back() : (path.back()-allowed_action_num);
                int act=(a<allowed_action_num) ? a : (a-allowed_action_num);
                if(to_group_id[prev]==to_group_id[act] && to_order_in_group[prev]>to_order_in_group[act])
                    continue;
                // より短い別の動作で置換できる場合
                if(path.back()==a)
                    next_same_action_num = same_action_num+1;
                if(2*next_same_action_num>to_rotate_num[act] 
                    || (2*next_same_action_num==to_rotate_num[act] && act!=a))
                    continue;
            }

            auto next=do_action(state, a);
            auto next_hash=zhash.hash(next);
            if (visit.contains(next_hash))
                continue;
            visit.emplace(next_hash);
            path.emplace_back(a);
            int t=search(next, path, g + 1, next_same_action_num);
            if (FOUND==t) return FOUND;
            if (t < minval) minval = t;
            path.pop_back();
            visit.erase(next_hash);
        }
        return minval;
    }
};

void check_answer(const string &filename){
    auto actions=load_actions(filename);
    auto result = simulation(initial_state, actions);
    int mistake = get_mistakes(result);
    assert(mistake<=num_wildcards);
}

// 状態
struct State {
    VI actions;
    // actions[i]適用結果はstates[i+1]に格納される
    VVI states;
    VI mistakes;
    double score;
    double annealing_score;

    State(){}
    void initialize(const VI &init_actions){
        if(DEBUG) OUT("initialize");
        // 初期解 ----------
        actions=init_actions;
        states.clear();
        mistakes.clear();
        states.emplace_back(initial_state);
        mistakes.emplace_back(get_mistakes(states.back()));
        simulate(0);
    }
    void simulate(int from){
        // 初期状態
        states.resize(from+1);
        mistakes.resize(from+1);
        FOR(i, from, SZ(actions)){
            int a = actions[i];
            states.emplace_back(do_action(states.back(), a));
            mistakes.emplace_back(get_mistakes(states.back()));
        }// ----------------
    }

    tuple<double, double> calc_score() {
        score = 0;
        annealing_score = 0;
        // スコア ----------
        score=actions.size();
        annealing_score=score + 100*max(0, mistakes.back()-num_wildcards);
        // -----------------
        return {score, annealing_score};
    }
    void print_answer() const {
        // 答え表示 ---------
        OUT(action_decode(actions));
        // -----------------
    }
    // ターンがあるような場合
    vector<State> next_states() const {
        // 未実装
        vector<State> next;
        return next;
    }
    bool operator<(const State& s) const {
        return score < s.score;
    }
    bool operator>(const State& s) const {
        return score > s.score;
    }
};

// 時間制限
constexpr int TIME_LIMIT = INF;


double annealing(const string &filename, ChronoTimer &timer, int loop_max, int verbose){
    assert(file_exists(filename));
    auto actions=load_actions(filename);
    int current_best=SZ(actions);
    dump(current_best)

    if(DEBUG) OUT("initial score:", current_best);
    IDAstar idastar;
    REP(loop, loop_max){
        if (DEBUG && loop % verbose == 0)
            OUT(loop, "\t:", SZ(actions));
        // 
        VVI states;
        states.emplace_back(initial_state);
        for(int a: actions)
            states.emplace_back(do_action(states.back(), a));

        // 操作
        int i=get_rand(SZ(states));
        int j=(i+get_rand(1,SZ(states)))%SZ(states);
        if(i>j)swap(i,j);
        int length=j-i;
        //
        dump(i)
        dump(j)
        dump(length)
        // auto path=search(states[i], states[j], length, false, states[j]==solution_state ? num_wildcards : 0);
        auto path=idastar.ida_star(states[i], states[j], length, states[j]==solution_state ? num_wildcards : 0);
        
        if(!path)continue;
        const auto &res = path.value();
        if(SZ(res)>=length)continue;
        REP(k, SZ(res)){
            actions[i]=res[k];
            ++i;
        }
        actions.erase(actions.begin()+i, actions.begin()+j);
        OUT("update!!!!!", SZ(actions));
        if(SZ(actions)<current_best){
            OUT("saved", current_best, "->", SZ(actions));
            save_actions(filename, actions);
            check_answer(filename);
            current_best=SZ(actions);
        }
    }
    return SZ(actions);
}

int ida_solve(const string &filename){
    assert(file_exists(filename));
    auto current_best=load_actions(filename);
    dump(SZ(current_best))

    IDAstar idastar;
    auto res=idastar.ida_star(initial_state, solution_state, SZ(current_best), num_wildcards);
    if(res){
        if(SZ(res.value())<SZ(current_best)){
            OUT("saved", SZ(current_best), "->", SZ(res.value()));
            save_actions(filename, res.value());
        }
        return SZ(res.value());
    }
    return INF;
}

const string DATA_DIR = "./data/";
int main() {
    ChronoTimer timer;
    int case_num = 398;
    double sum_score = 0.0;
    double sum_log_score = 0.0;
    int64_t max_time = 0;
    REP(i, case_num){
    // FOR(i, 30, case_num){
    // FOR(i, 284, case_num){
    // FOR(i, 338, case_num){
        timer.start();
        dump(SEED)
        rand_engine.seed(SEED);
        OUT("data_load");
        string input_filename = to_string(i) + ".txt";
        string file_path = DATA_DIR + input_filename;
        ifstream ifs(file_path);
        assert(!ifs.fail());
        data_load(ifs);

        string output_filename="output/"+to_string(i)+".txt";
        double score=ida_solve(output_filename);
        // double score = annealing(output_filename, timer, 100, 100);
        // double score = search(output_filename, false, 1);
        // double score = search(output_filename, true, 1);
        // double score = search(output_filename, false, 0.95);
        // double score = search(output_filename, true, 0.95);
        // double score = search(output_filename, false, 0.01);
        check_answer(output_filename);
        timer.end();
        if(DEBUG) {
            auto time = timer.time();
            sum_score += score;
            sum_log_score += log1p(score);
            max_time = max(max_time, time);
            // dump(initial_state)
            // dump(solution_state)
            // dump(allowed_moves_name)
            // dump(allowed_moves)
            OUT("--------------------");
            OUT("case_num: ", i);
            // OUT("puzzle_type: ", puzzle_type);
            // OUT("state_length: ", state_length);
            // OUT("allowed_action_num: ", allowed_action_num);
            // OUT("num_wildcards: ", num_wildcards);
            OUT("score: ", score);
            OUT("time: ", time);
            // OUT("mean_score: ", sum_score/(i+1));
            // OUT("mean_log_score: ", sum_log_score/(i+1));
            OUT("sum_score: ", sum_score);
            OUT("sum_log_score: ", sum_log_score);
            OUT("max_time: ", max_time);
            OUT("--------------------");
        }
    }
    return 0;
}
