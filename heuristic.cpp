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
    // for ( const T& item : vec ) os << item << ",";
    return os;
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
const int SEED = random_device()();
// constexpr int SEED = 1;
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
string kind; int X,Y;
int state_length;
map<string, int> label_mapping;
VI initial_state;
VI solution_state;
int allowed_action_num;
VS allowed_moves_name;
// VVI allowed_moves;
vector<unordered_map<int,int>> allowed_moves;
int num_wildcards;
// 逆操作
// VVI allowed_moves_inverse;
vector<unordered_map<int,int>> allowed_moves_inverse;
// vector<PII> same_color_index;

#define get_action(id) ((id)<allowed_action_num ? allowed_moves[id] : allowed_moves_inverse[(id)-allowed_action_num])
#define get_action_name(id) ((id)<allowed_action_num ? allowed_moves_name[id] : ("-"+allowed_moves_name[(id)-allowed_action_num]))

VI do_action(const VI& state, int action_id){
    auto s=state;
    const auto &action = get_action(action_id);
    // REP(i, state_length)
    //     s[i]=state[action[i]];
    for(auto [i, v]: action)
        s[i]=state[v];
    return s;
}

VI simulation(const VI& state, const VI &actions){
    auto s=state;
    for(int a: actions)
        s = do_action(s, a);
    return s;
}

unordered_map<int,int> inverse(const unordered_map<int,int> &move){
    unordered_map<int,int> inv;
    for(auto[i,v]:  move)
        inv[v]=i;
    return inv;
}
VI inverse(const VI &move){
    auto inv=move;
    REP(i, SZ(inv))
        inv[move[i]]=i;
    return inv;
}
int inverse(int a){
    return a<allowed_action_num ? a+allowed_action_num : a-allowed_action_num;
}

// 
VI to_group_id;//(allowed_action_num);
VI to_order_in_group;//(allowed_action_num);
VVI group;

void devide_independent_action(){
    vector<SI> changes(allowed_action_num);
    REP(i, allowed_action_num){
        const auto& move=allowed_moves[i];
        // REP(j, SZ(move))if(j!=move[j]){
        //     changes[i].emplace(j);
        //     changes[i].emplace(move[j]);
        // }
        for(auto [j, v] : move){
            assert(j!=v);
            changes[i].emplace(j);
            changes[i].emplace(v);
        }
    }
    to_group_id.assign(allowed_action_num, 0);
    to_order_in_group.assign(allowed_action_num, 0);
    group.clear();
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
VI group_to_rotate_num;
void inv_operation(){
    to_rotate_num.assign(allowed_action_num, 0);
    group_to_rotate_num.assign(group.size(), 0);
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
        assert(group_to_rotate_num[to_group_id[a]]==0 || (group_to_rotate_num[to_group_id[a]] == to_rotate_num[a]));
        group_to_rotate_num[to_group_id[a]] = to_rotate_num[a];
    }
    dump(to_rotate_num)
}

// VVI to_goal_step;
// VVI to_start_step;
// VVI pre_heuristic(const VI& target_state){
//     VVI result(state_length, VI(SZ(label_mapping), INF));
//     queue<tuple<VI, int, int, int>> que;
//     que.emplace(target_state, 0, -1, 0);
//     while(!que.empty()){
//         auto[state,cnt,prev_a,same_action_num]=que.front();que.pop();
//         bool fin=true;
//         REP(i, state_length){
//             result[i][state[i]]=min(result[i][state[i]], cnt);
//             REP(j, SZ(label_mapping))if(result[i][j]==INF){
//                 fin=false;
//                 break;
//             }
//         }
//         if(fin) break;
//         REP(a, allowed_action_num*2){
//             int next_same_action_num=1;
//             if(prev_a>=0){
//                 if(a+allowed_action_num==prev_a || a==prev_a+allowed_action_num)
//                     continue;
//                 // グループ順序
//                 int prev=(prev_a<allowed_action_num) ? prev_a : (prev_a-allowed_action_num);
//                 int act=(a<allowed_action_num) ? a : (a-allowed_action_num);
//                 if(to_group_id[prev]==to_group_id[act] && to_order_in_group[prev]>to_order_in_group[act])
//                     continue;
//                 // より短い別の動作で置換できる場合
//                 if(prev_a==a)
//                     next_same_action_num = same_action_num+1;
//                 if(2*next_same_action_num>to_rotate_num[act] 
//                     || (2*next_same_action_num==to_rotate_num[act] && act!=a))
//                     continue;
//             }
//             que.emplace(do_action(state, a), cnt+1, a, next_same_action_num);
//         }
//     }
//     return result;
// }
const vector<PII> DIRECT{
    {-1,0},
    {1,0},
    {0,1},
    {0,1},
};


vector<PII> same_color_index_pair(){
    vector<PII> index_pair;
    if(kind=="cube"){
        REP(k, 6)REP(i, X)REP(j, X){
            int idx=k*X*X+i*X+j;
            int c=solution_state[idx];
            for(auto [di,dj]: DIRECT){
                int i2=i+di, j2=j+dj;
                if(i2<0||j2<0||i2>=X||j2>=X)continue;
                int idx2=k*X*X+i2*X+j2;
                if(idx>=idx2)continue;
                int c2=solution_state[idx2];
                if(c==c2)
                    index_pair.emplace_back(idx, idx2);
            }
        }
    }else if(kind=="wreath"){
        // not implement
    }else if(kind=="globe"){
        // 
        REP(i, X+1)REP(j, 2*Y){
            int idx=i*2*Y+j;
            int c=solution_state[idx];
            for(auto [di,dj]: DIRECT){
                int i2=i+di, j2=j+dj;
                if(i2<0||j2<0||i2>=X||j2>=X)continue;
                int idx2=i2*2*Y+j2;
                if(idx>=idx2)continue;
                int c2=solution_state[idx2];
                if(c==c2)
                    index_pair.emplace_back(idx, idx2);
            }
        }
    }else
        assert(false);
    return index_pair;
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
    // allowed_moves.assign(allowed_action_num, VI(state_length));
    allowed_moves.assign(allowed_action_num, unordered_map<int,int>());
    REP(i, allowed_action_num){
        is  >> allowed_moves_name[i];
            // >> allowed_moves[i];
        REP(j, state_length){
            int v;is>>v;
            if(v!=j)
                allowed_moves[i][j]=v;
        }
    }
    is >> num_wildcards;
    // 逆操作
    // allowed_moves_inverse.assign(allowed_action_num, VI());
    allowed_moves_inverse.assign(allowed_action_num, unordered_map<int,int>());
    REP(i, allowed_action_num){
        allowed_moves_inverse[i]=inverse(allowed_moves[i]);
    }
    // グループ分け
    devide_independent_action();
    // 回転数
    inv_operation();
    
    auto tmp=split_str(puzzle_type, '_', true);
    kind=tmp[0];
    tmp=split_str(tmp[1], '/', true);
    X = stoi(tmp[0]);
    Y = stoi(tmp[1]);

    dump(puzzle_type)
    dump(num_wildcards)
    dump(allowed_action_num)
    // 
    // to_goal_step=pre_heuristic(solution_state);
    // to_start_step=pre_heuristic(initial_state);
    // 
    // same_color_index=same_color_index_pair();
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

// 同じ色
// int get_penalty(const VI& state){
//     int cnt=0;
//     for(auto[i,j]:same_color_index)
//         cnt += state[i]!=state[j];
//     return cnt;
// }

string action_decode(const VI& actions, const string & delim="."){
    string ans="";
    REP(i, SZ(actions)){
        int a=actions[i];
        ans += get_action_name(a);
        if(i+1!=SZ(actions))
            ans += delim;
    }
    return ans;
}
VI action_encode(const string& str, char delim='.'){
    map<string, int> action_mapping;
    REP(i, allowed_action_num){
        action_mapping[allowed_moves_name[i]]=i;
        action_mapping["-"+allowed_moves_name[i]]=i+allowed_action_num;
    }

    VI actions;
    for(auto& a: split_str(str, delim, true))
        actions.emplace_back(action_mapping[a]);
    return actions;
}
// 逆操作にする
VI inverse_action(const VI& path){
    auto action=path;
    for(int &a:action)
        a = (a < allowed_action_num) ? (a+allowed_action_num): (a-allowed_action_num);
    REVERSE(action);
    return action;
}


VI load_actions(const string &filename){
    ifstream ifs(filename);
    string str;
    ifs>>str;
    return action_encode(str);
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
        if(a>=0)actions.emplace_back(a);
        h=next;
    }
    REVERSE(actions);
    return actions;
}

// pqで状態が近い順に探索する
optional<VI> search(const VI &start_state, const VI& goal_state, int current_best_size, bool strong_search, int wildcards=0, size_t give_up=100000000){
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

// pqで状態が近い順に探索する
int search(const string &output, bool strong_search, double improve_rate=1){
    // 保存してあるベスト解
    int current_best_size=INF;
    if(file_exists(output)){
        current_best_size=SZ(load_actions(output));
        dump(current_best_size)
    }
    auto res=search(initial_state, solution_state, current_best_size, strong_search, num_wildcards);
    if (res){
        auto &actions=res.value();
        int mistake = get_mistakes(simulation(initial_state, actions));
        assert(mistake<=num_wildcards);
        OUT("solved");
        if(SZ(actions)<current_best_size){
            OUT("saved", current_best_size, "->", SZ(actions));
            save_actions(output, actions);
        }
        return SZ(actions);
    }
    return INF;
}

enum{
    FOUND,
    OUTSIDE_FOUND,
    FAIL
};

struct IDAstar{
    ZobristHashing<uint64_t> zhash;
    int max_action_size;
    // int bound; 
    double bound;
    VI actions;

    // 一方向探索時の変数
    unordered_set<uint64_t> visit;
    VI goal_state;
    int wildcards;
    int searched;

    IDAstar():zhash(SZ(label_mapping), state_length, rand_engine){}
    double h(const VI& state)const{
        return max(0.0, (get_mistakes(state, goal_state)-wildcards)/(double)state_length);
    }
    double h(const VI& state, const VI& state_g, int wild)const{
        // return max(0, get_mistakes(state, state_g)-wild);
        return max(0.0, (get_mistakes(state, state_g)-wild)/(double)state_length);
    }
    // 一方方向のとき
    int h_int(const VI& state)const{
        return max(0, get_mistakes(state, goal_state)-wildcards);
        // return max(0, get_mistakes(state, goal_state)-wildcards)?1:0;
    }
    int h_int(const VI& state, const VI& state_g, int wild)const{
        return max(0, get_mistakes(state, state_g)-wild);
    }
    optional<VI> ida_star(const VI &init_state, const VI &goal_state, int max_action_size, int wildcards){
        this->wildcards=wildcards;
        this->goal_state=goal_state;
        this->max_action_size=max_action_size;
        this->bound = h_int(init_state);
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
            dump(searched)
        }
    }
    optional<VI> ida_star_bidirect(const VI &start_state, const VI &goal_state, int max_action_size, int wild){
        this->max_action_size=max_action_size;
        this->bound = h(start_state, goal_state, wild);
        // this->bound = INF;

        auto hash_s=zhash.hash(start_state);
        auto hash_g=zhash.hash(goal_state);
        stack<tuple<int, int, uint64_t>> st_s, st_g;
        // queue<tuple<int, int, uint64_t>> st_s, st_g;
        // MINPQ<tuple<double, int, int, int, uint64_t>> st_s, st_g;
        st_s.emplace(0, 0, hash_s);
        st_g.emplace(0, 0, hash_g);
        // st_s.emplace(hash_s, 0, 0, 0, hash_s);
        // st_g.emplace(hash_g, 0, 0, 0, hash_g);

        unordered_map<uint64_t, VI> frontier_s, frontier_g;
        frontier_s[hash_s]=VI();
        frontier_g[hash_g]=VI();

        // MAXPQ<pair<double, uint64_t>> score_s, score_g;
        unordered_map<uint64_t, int> outside_s, outside_g;

        while(1){
            dump(bound)
            stack<tuple<int, int, uint64_t>> next_st_s, next_st_g;
            // queue<tuple<int, int, uint64_t>> next_st_s, next_st_g;
            // MINPQ<tuple<double, int, int, int, uint64_t>> next_st_s, next_st_g;
            while(!st_s.empty() || !st_g.empty()){
                if(!st_s.empty()){
                    // if(FOUND==next_node(st_s, next_st_s, frontier_s, score_s, start_state, goal_state, wild, frontier_g, true))
                    // if(FOUND==next_node(st_s, next_st_s, frontier_s, start_state, goal_state, wild, frontier_g, true))
                    int r=next_node(st_s, next_st_s, frontier_s, start_state, goal_state, wild, frontier_g, outside_g, true, 0);
                    if(r>=0)return actions;
                }
                if(!st_g.empty()){
                    // if(FOUND==next_node(st_g, next_st_g, frontier_g, score_g, goal_state, start_state, 0, frontier_s, false))
                    // if(FOUND==next_node(st_g, next_st_g, frontier_g, goal_state, start_state, 0, frontier_s, false))
                    int r=next_node(st_g, next_st_g, frontier_g, goal_state, start_state, 0, frontier_s, outside_s, false, 0);
                    if(r>=0) return actions;
                }
            }
            dump(next_st_s.size())
            dump(frontier_s.size())
            // st_s=next_st_s, st_g=next_st_g;
            while(!next_st_s.empty()){
                st_s.emplace(next_st_s.top());
                next_st_s.pop();
            }
            while(!next_st_g.empty()){
                st_g.emplace(next_st_g.top());
                next_st_g.pop();
            }
            bound++;
            if(bound>=max_action_size*2)
                break;
        }
        return nullopt;
    }
    
    optional<tuple<VI, int, int>> ida_star_bidirect(const VVI &states, int i, int j){
        int length=j-i;
        auto &start_state=states[i], &goal_state=states[j];
        int wild = states[j]==solution_state ? num_wildcards : 0;

        unordered_map<uint64_t, int> outside_s, outside_g;
        REP(k, SZ(states)){
            if(k<i) outside_s[zhash.hash(states[k])]=k;
            else if(k>j) outside_g[zhash.hash(states[k])]=k;
        }

        this->max_action_size=(length+1)/2;
        this->bound = h(start_state, goal_state, wild);

        auto hash_s=zhash.hash(start_state);
        auto hash_g=zhash.hash(goal_state);
        stack<tuple<int, int, uint64_t>> st_s, st_g;
        st_s.emplace(0, 0, hash_s);
        st_g.emplace(0, 0, hash_g);

        unordered_map<uint64_t, VI> frontier_s, frontier_g;
        frontier_s[hash_s]=VI();
        frontier_g[hash_g]=VI();

        while(1){
            dump(bound)
            stack<tuple<int, int, uint64_t>> next_st_s, next_st_g;
            while(!st_s.empty() || !st_g.empty()){
                if(!st_s.empty()){
                    int r=next_node(st_s, next_st_s, frontier_s, start_state, goal_state, wild, frontier_g, outside_g, true, j);
                    if(r>=0)  return tuple{actions, i, r};
                }
                if(!st_g.empty()){
                    int r=next_node(st_g, next_st_g, frontier_g, goal_state, start_state, 0, frontier_s, outside_s, false, i);
                    if(r>=0)  return tuple{actions, r, j};
                }
            }
            dump(next_st_s.size())
            dump(frontier_s.size())
            while(!next_st_s.empty()){
                st_s.emplace(next_st_s.top());
                next_st_s.pop();
            }
            while(!next_st_g.empty()){
                st_g.emplace(next_st_g.top());
                next_st_g.pop();
            }
            bound++;
            if(bound>=max_action_size*2)
                break;
        }
        return nullopt;
    }
    int next_node(
        stack<tuple<int, int, uint64_t>> &st,
        stack<tuple<int, int, uint64_t>> &next_st,
        // queue<tuple<int, int, uint64_t>> &st,
        // queue<tuple<int, int, uint64_t>> &next_st,
        // MINPQ<tuple<double, int, int, int, uint64_t>> &st,
        // MINPQ<tuple<double, int, int, int, uint64_t>> &next_st,
        unordered_map<uint64_t, VI>& frontier,
        // MAXPQ<pair<double, uint64_t>> &score,
        const VI& init_state,
        const VI& target_state, 
        int wild, 
        unordered_map<uint64_t, VI>& frontier_target, 
        unordered_map<uint64_t, int>& outside,
        bool from_start,
        int index
    ){
        auto[a, same_action_num, hash]=st.top();st.pop();
        // auto[a, same_action_num, hash]=st.front();st.pop();
        // auto[_, __, a, same_action_num, hash]=st.top();st.pop();
        if(a+1<allowed_action_num*2)
            st.emplace(a+1, same_action_num, hash);
            // st.emplace(_, __, a+1, same_action_num, hash);
        assert(frontier.contains(hash));
        // if(!frontier.contains(hash))
        //     return INF;
        auto &path=frontier[hash];
        // auto path=frontier[hash];
        int next_same_action_num=1;
        if(!path.empty()){
            // 逆操作
            if(a+allowed_action_num==path.back() || a==path.back()+allowed_action_num)
                return -1;
            // グループ順序
            int prev=(path.back()<allowed_action_num) ? path.back() : (path.back()-allowed_action_num);
            int act=(a<allowed_action_num) ? a : (a-allowed_action_num);
            if(to_group_id[prev]==to_group_id[act] && to_order_in_group[prev]>to_order_in_group[act])
                return -1;
            // より短い別の動作で置換できる場合
            if(path.back()==a)
                next_same_action_num = same_action_num+1;
            if(2*next_same_action_num>to_rotate_num[act] 
                || (2*next_same_action_num==to_rotate_num[act] && act!=a))
                return -1;
        }
        path.emplace_back(a);
        auto next = simulation(init_state, path);
        auto next_hash=zhash.hash(next);
        if(frontier.contains(next_hash) && SZ(frontier[next_hash])<=SZ(path)){
            path.pop_back();
            return -1;
        }
        frontier[next_hash]=path;

        auto h_value=h(next, target_state, wild);
        if(h_value==0 || outside.contains(next_hash)){
            OUT("find0!!!");
            this->actions = path;
            if(!from_start)
                this->actions=inverse_action(this->actions);
            path.pop_back();
            return h_value==0 ? index : outside[next_hash];
        }
        if(frontier_target.contains(next_hash)){
            if(from_start){
                OUT("find1!!!");
                auto action=inverse_action(frontier_target[next_hash]);
                this->actions=path;
                this->actions.insert(this->actions.end(), action.begin(), action.end());
            }else{
                OUT("find2!!!");
                auto action=inverse_action(path);
                this->actions=frontier_target[next_hash];
                this->actions.insert(this->actions.end(), action.begin(), action.end());
            }
            path.pop_back();
            return index;
        }
        // score.emplace(h_value, next_hash);
        // if(score.size()>10000000L){
        //     auto[_, maxhash]=score.top();score.pop();
        //     frontier.erase(maxhash);
        // }
        auto f = SZ(path) + h_value;
        if (f <= bound && SZ(path) < this->max_action_size){
            st.emplace(0, next_same_action_num, next_hash);
            // st.emplace(h_value, SZ(path), 0, next_same_action_num, next_hash);
        }else{
            next_st.emplace(0, next_same_action_num, next_hash);
        }
        path.pop_back();
        return -1;
    }
    int search(VI &state, VI &path, int g, int same_action_num){
        ++searched;
        int f = g + h_int(state);
        if (f > bound) return f;
        if (f==g){
            actions=path;
            return FOUND;
        }
        int minval=INF;
        if(g >= max_action_size) return minval;
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

struct RandomWalk{
    ZobristHashing<uint64_t> zhash;
    unordered_set<uint64_t> visit;
    VVI data;
    int maxdepth;
    unordered_set<uint64_t> data_hash;

    RandomWalk(int d): zhash(SZ(label_mapping), state_length, rand_engine), maxdepth(d){}
    void make_dataset(VI &state, int size){
        auto hash=zhash.hash(state);
        REP(i, size){
            VI path;
            visit.emplace(hash);
            random_walk(state, path, 0, hash);
            visit.clear();
            if(i%10000==0)
                dump(SZ(data))
        }
    }
    bool random_walk(const VI &state, VI &path, int same_action_num, uint64_t hash){
        if (SZ(path) >= maxdepth) {
            if(data_hash.contains(hash))
                return false;
            data.emplace_back(path);
            data_hash.emplace(hash);
            return true;
        }
        REP(_, 100000){
            int a= get_rand(allowed_action_num*2);
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
            if(random_walk(next, path, next_same_action_num, next_hash))
                return true;
            path.pop_back();
            visit.erase(next_hash);
        }
        return false;
    }

    void save_data(const string &filename){
        ofstream ofs(filename);
        visit.clear();
        visit.emplace(zhash.hash(solution_state));
        ofs<<0<<","<<solution_state<<endl;
        for (auto & actions: data){
            auto s=solution_state;
            int cnt=0;
            for(int a: actions){
                s = do_action(s, a);
                ++cnt;
                auto hash=zhash.hash(s);
                if(visit.contains(hash))
                    continue;
                ofs<<cnt<<","<<s<<endl;
                visit.emplace(hash);
            }
        }
        ofs.close();
    }
};

// 保存した解のチェック
int check_answer(const string &filename){
    auto actions=load_actions(filename);
    auto result = simulation(initial_state, actions);
    int mistake = get_mistakes(result);
    assert(mistake<=num_wildcards);
    return SZ(actions);
}

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
        // int width=1+get_rand(min(SZ(states)-1, 30));
        // int width=1+get_rand(11, min(SZ(states)-1, 28));
        int width=1+get_rand(11, min(SZ(states)-1, 24));
        // int width=1+get_rand(11, min(SZ(states)-1, 22));
        // int width=1+get_rand(min(SZ(states)-1, 20));
        // int width=1+get_rand(min(SZ(states)-1, 14));
        // int width=1+get_rand(min(SZ(states)-1, 12));
        // int width=1+get_rand(min(SZ(states)-1, 10));
        // int width=1+get_rand(min(SZ(states)-1, 8));
        // int width=1+get_rand(min(SZ(states)-1, 6));
        int i=get_rand(SZ(states)-width);
        int j=i+width;
        if(i>j)swap(i,j);
        int length=j-i;
        //
        dump(i)
        dump(j)
        dump(length)
        // auto path=search(states[i], states[j], length, false, states[j]==solution_state ? num_wildcards : 0);
        // auto path=idastar.ida_star(states[i], states[j], length, states[j]==solution_state ? num_wildcards : 0);
        // auto path=idastar.ida_star_bidirect(states[i], states[j], length, states[j]==solution_state ? num_wildcards : 0);
        // auto path=idastar.ida_star_bidirect(states[i], states[j], (length+1)/2, states[j]==solution_state ? num_wildcards : 0);
        auto res=idastar.ida_star_bidirect(states, i, j);
        if(!res)continue;
        VI path;
        tie(path, i, j)=res.value();
        length=j-i;
        if(SZ(path)>=length)continue;
        REP(k, SZ(path)){
            actions[i]=path[k];
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

// 同じ状態が現れたら区間をスキップする
VI same_state_skip(const VI& action){
    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);
    unordered_map<uint64_t, int> hash_to_idx;
    unordered_map<int, uint64_t> idx_to_hash;
    auto state=initial_state;
    auto h=zhash.hash(state);
    hash_to_idx[h]=0;
    idx_to_hash[0]=h;
    VI result;
    for(int a: action){
        state=do_action(state,a);
        auto h=zhash.hash(state);
        if (hash_to_idx.contains(h)){
            for(int j=hash_to_idx[h]+1; j<=SZ(result); ++j){
                hash_to_idx.erase(idx_to_hash[j]);
                idx_to_hash.erase(j);
            }
            result.resize(hash_to_idx[h]);
        }else{
            result.emplace_back(a);
            hash_to_idx[h]=SZ(result);
            idx_to_hash[SZ(result)]=h;
        }
    }
    return result;
}

// 短い演算に置き換える
// aaa → -a
VI loop_compress(const VI& action){
    int prev=-1, cnt=0;
    VI result;
    auto state=initial_state;
    for(int a:action){
        state=do_action(state,a);
        if(prev!=a)cnt=1;
        else cnt++;
        result.emplace_back(a);
        int act=a < allowed_action_num ? a : (a-allowed_action_num);
        // assert(cnt <= to_rotate_num[act]/2);
        if(cnt > to_rotate_num[act]/2){
            result.resize(SZ(result)-cnt);
            int a_inv=a < allowed_action_num ? (a+allowed_action_num) : (a-allowed_action_num);
            REP(_, to_rotate_num[act]-cnt)
                result.emplace_back(a_inv);
            prev=a_inv;
        }else
            prev=a;
    }
    return result;
}

// 途中でゴール条件を満たすか確認する
VI wildcard_finish(const VI& action){
    VI result;
    auto state=initial_state;
    for(int a:action){
        state=do_action(state,a);
        result.emplace_back(a);
        int mistake = get_mistakes(state);
        if(mistake<=num_wildcards)
            break;
    }
    return result;
}


int ida_solve(const string &filename, bool dual){
    int current_best_size=INF;
    if(file_exists(filename)){
        auto current_best=load_actions(filename);
        current_best_size=SZ(current_best);
        dump(current_best_size)
    }

    IDAstar idastar;
    auto res = dual ?
        //   idastar.ida_star_bidirect(initial_state, solution_state, SZ(current_best), num_wildcards)
        // NOTE: wildcardがあるときは深さ制限すると現在よりも悪化する解しか見つからず失敗する可能性がある
          idastar.ida_star_bidirect(initial_state, solution_state, (current_best_size+1)/2, num_wildcards)
        : idastar.ida_star(initial_state, solution_state, current_best_size, num_wildcards);
    
    if(res){
        if(SZ(res.value())<current_best_size){
            OUT("saved", current_best_size, "->", SZ(res.value()));
            save_actions(filename, res.value());
        }
        return SZ(res.value());
    }
    return INF;
}


void make_dataset(int i){
    RandomWalk rw(50);
    rw.make_dataset(solution_state, 10000);
    rw.save_data("nn/"+to_string(i)+".csv");
}

double random_walk(int i){
    RandomWalk rw_s(100), rw_g(100);
    rw_s.make_dataset(initial_state, 100000);
    rw_g.make_dataset(solution_state, 100000);

    unordered_set<uint64_t> visit_s;
    visit_s.emplace(rw_s.zhash.hash(initial_state));
    for (auto & actions: rw_s.data){
        auto s=initial_state;
        for(int a: actions){
            s = do_action(s, a);
            visit_s.emplace(rw_s.zhash.hash(s));
        }
    }
    assert(!visit_s.contains(rw_s.zhash.hash(solution_state)));
    for (auto & actions: rw_g.data){
        auto s=solution_state;
        for(int a: actions){
            s = do_action(s, a);
            assert(!visit_s.contains(rw_s.zhash.hash(s)));
        }
    }
    return 0;
}

#pragma omp declare reduction(maximum : PII : omp_out=(omp_out>omp_in ? omp_out : omp_in)) initializer(omp_priv = omp_orig)
VI greedy_improve(const VI &action, int depth){
    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);
    unordered_map<uint64_t, int> hash_to_idx;
    
    auto state=initial_state;
    int idx=0;
    hash_to_idx[zhash.hash(state)]=idx++;
    for(int a: action){
        state = do_action(state, a);
        hash_to_idx[zhash.hash(state)]=idx++;
    }
    const int total_actions=allowed_action_num*2;

    VI result;
    state=initial_state;
    for(idx=0; idx<SZ(action); ){
        PII max_idx_act{-1,-1};
        PII max_idx_act2{-1,-1};
        // 近傍をチェックして同じ状態があればジャンプする
        #pragma omp parallel for reduction(maximum: max_idx_act, max_idx_act2)
        REP(a, total_actions){
            auto new_state = do_action(state, a);
            auto new_hash = zhash.hash(new_state);
            if (hash_to_idx.contains(new_hash)){
                int tmp_idx = hash_to_idx[new_hash];
                if (tmp_idx > max_idx_act.first){
                    max_idx_act.first=tmp_idx;
                    max_idx_act.second = a;
                }
            }
            if(num_wildcards!=0 && get_mistakes(new_state)<=num_wildcards){
                max_idx_act.first=INF;
                max_idx_act.second = a;
                // break;
            }else if(depth>=2){
                REP(a2, total_actions){
                    auto new_state2 = do_action(new_state, a2);
                    auto new_hash2 = zhash.hash(new_state2);
                    if (hash_to_idx.contains(new_hash2)){
                        int tmp_idx = hash_to_idx[new_hash2];
                        if (tmp_idx > max_idx_act2.first){
                            max_idx_act2.first=tmp_idx;
                            max_idx_act2.second=a*total_actions + a2;
                        }
                    }
                    if(num_wildcards!=0 && get_mistakes(new_state2)<=num_wildcards){
                        max_idx_act2.first=INF;
                        max_idx_act2.second=a*total_actions + a2;
                        // break;
                    }
                }
            }
        }
        assert(max_idx_act.first>idx);
        if(max_idx_act.first==INF){
            result.emplace_back(max_idx_act.second);
            break;
        }else if(max_idx_act2.first==INF){
            int a=max_idx_act2.second/total_actions, a2=max_idx_act2.second%total_actions;
            result.emplace_back(a);
            result.emplace_back(a2);
            break;
        }else if(max_idx_act2.first > max_idx_act.first+1){
            idx = max_idx_act2.first;
            int a=max_idx_act2.second/total_actions, a2=max_idx_act2.second%total_actions;
            result.emplace_back(a);
            result.emplace_back(a2);
            state = do_action(state, a);
            state = do_action(state, a2);
        }else{
            idx = max_idx_act.first;
            result.emplace_back(max_idx_act.second);
            state = do_action(state, max_idx_act.second);
        }
    }
    return result;
}

void dfs(
    const VI &state, VI &path, int same_action_num, size_t maxdepth, 
    ZobristHashing<uint64_t> &zhash, unordered_map<uint64_t, tuple<int,uint64_t,int>>& visit, 
    uint64_t prev_hash, pair<int, uint64_t> &to_end,
    int random_prune=0
){
    if (path.size() > maxdepth) return ;
    auto hash=zhash.hash(state);
    if (visit.contains(hash) && get<0>(visit[hash])<=SZ(path))
        return;
    visit[hash]={SZ(path), prev_hash, path.empty()? -1: path.back()};
    if(num_wildcards!=0 && SZ(path)<to_end.first && get_mistakes(state)<=num_wildcards)
        to_end={SZ(path), hash};
    REP(a, allowed_action_num*2){
        if (random_prune && get_rand(100)<random_prune)
            continue;
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
        path.emplace_back(a);
        dfs(do_action(state, a), path, next_same_action_num, maxdepth, zhash, visit, hash, to_end, random_prune);
        path.pop_back();
    }
}

#pragma omp declare reduction(maximum_tuple : tuple<int,int,VI> : omp_out=(omp_out>omp_in ? omp_out : omp_in)) initializer(omp_priv = omp_orig)
VI dual_greedy_improve(const VI &action, int depth, int search_step=INF, int random_prune=0){
    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);

    VVI states{initial_state};
    auto state=initial_state;
    for(int a: action){
        state = do_action(state, a);
        states.emplace_back(state);
    }

    vector<unordered_map<uint64_t, tuple<int,uint64_t,int>>> hash_to_depth(SZ(states));
    vector<pair<int, uint64_t>> to_end(SZ(states), {INF, 0});
    #pragma omp parallel for
    REP(i, SZ(states)){
        unordered_map<uint64_t, tuple<int,uint64_t,int>> visit; VI path;
        dfs(states[i], path, 0, depth, zhash, visit, 0, to_end[i], random_prune);
        hash_to_depth[i]=visit;
        dump(i)
        dump(hash_to_depth[i].size())
    }

    VI result;
    for(int i=0; i<SZ(action); ){
        dump(i)
        // 近傍をチェックして同じ状態があればジャンプする
        tuple<int,int,VI> improve_idx_act={0,i+1,VI{action[i]}};
        // ワイルドカードのゴールへ早く付く場合
        int imp=SZ(action) - to_end[i].first - i;
        if(imp>0){
            auto act=construct_actions(0, to_end[i].second, hash_to_depth[i]);
            dump(imp)
            result.insert(result.end(), act.begin(), act.end());
            break;
        }
        int end=min(SZ(action), i+search_step);
        #pragma omp parallel for reduction(maximum_tuple: improve_idx_act)
        for(int j=i+1; j<=end; ++j){
            for(auto [hash1, value1]: hash_to_depth[i]){
                auto it2=hash_to_depth[j].find(hash1);
                if(it2==hash_to_depth[j].end()) continue;
                int d1 = get<0>(value1);
                int d2 = get<0>(it2->second);
                int imp = (j-i) - (d1+d2);
                if(get<0>(improve_idx_act)<imp){
                    auto act=construct_actions(0, hash1, hash_to_depth[i]);
                    auto act2=inverse_action(construct_actions(0, it2->first, hash_to_depth[j]));
                    act.insert(act.end(), act2.begin(), act2.end());
                    improve_idx_act={imp, j, act};
                }
            }
        }
        auto &[improve, idx, act]=improve_idx_act;
        if(improve>0)
            dump(improve)
        i=idx;
        result.insert(result.end(), act.begin(), act.end());
    }
    return result;
}

VI dual_greedy_improve_low_memory(const VI &action, int depth, int search_step, int random_prune=0){
    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);

    VVI states{initial_state};
    auto state=initial_state;
    for(int a: action){
        state = do_action(state, a);
        states.emplace_back(state);
    }

    VI result;
    for(int i=0; i<SZ(action); ){
        dump(i)
        tuple<int,int,VI> improve_idx_act={0,i+1,VI{action[i]}};
        VI path;
        unordered_map<uint64_t, tuple<int,uint64_t,int>> visit1;
        pair<int, uint64_t> to_end{INF,0};
        dfs(states[i], path, 0, depth, zhash, visit1, 0, to_end, random_prune);
        dump(visit1.size())
        int imp=SZ(action) - to_end.first - i;
        if(imp>0){
            auto act=construct_actions(0, to_end.second, visit1);
            dump(imp)
            result.insert(result.end(), act.begin(), act.end());
            break;
        }

        int end=min(search_step, SZ(action));
        #pragma omp parallel for reduction(maximum_tuple: improve_idx_act)
        for(int j=i+1; j<=end; ++j){
            VI path;
            unordered_map<uint64_t, tuple<int,uint64_t,int>> visit2;
            pair<int, uint64_t> to_end{INF,0};
            dfs(states[j], path, 0, depth, zhash, visit2, 0, to_end, random_prune);

            for(auto [hash1, value1]: visit1){
                auto it2=visit2.find(hash1);
                if(it2==visit2.end()) continue;
                int d1 = get<0>(value1);
                int d2 = get<0>(it2->second);
                int imp = (j-i) - (d1+d2);
                if(get<0>(improve_idx_act)<imp){
                    auto act=construct_actions(0, hash1, visit1);
                    auto act2=inverse_action(construct_actions(0, it2->first, visit2));
                    act.insert(act.end(), act2.begin(), act2.end());
                    improve_idx_act={imp, j, act};
                }
            }
        }
        auto &[improve, idx, act]=improve_idx_act;
        if(improve>0)
            dump(improve)
        i=idx;
        result.insert(result.end(), act.begin(), act.end());
    }
    return result;
}

// // 良いと思った方向に進みまくる
// optional<VI> best_search(
//     const VI& state, int current_best_size, ZobristHashing<uint64_t> &zhash,
//     VI &actions, unordered_set<uint64_t> &visit, long &count
// ){
//     if(SZ(actions)>=current_best_size)
//         return nullopt;
//     if(++count%100000==0)
//         dump(count)

//     // mistake, act, hash
//     MINPQ<tuple<int, int, uint64_t>> candidate;
//     REP(a, allowed_action_num*2){
//         // 逆操作
//         if(!actions.empty() && (a+allowed_action_num==actions.back() || a==actions.back()+allowed_action_num))
//             continue;
//         auto next = do_action(state, a);
//         auto next_hash = zhash.hash(next);
//         if(visit.contains(next_hash))
//             continue;
//         int mis=get_mistakes(state);
//         if(mis<=num_wildcards){
//             actions.emplace_back(a);
//             return actions;
//         }
//         candidate.emplace(mis, a, next_hash);
//     }

//     while(!candidate.empty()){
//         auto[_,a,hash]=candidate.top();candidate.pop();
//         actions.emplace_back(a);
//         visit.emplace(hash);
//         // if(visit.size()%100000==0)
//         //     dump(visit.size())
//         auto res=best_search(do_action(state, a), current_best_size, zhash, actions, visit, count);
//         visit.erase(hash);
//         if(res)return res;
//         actions.pop_back();
//     }
//     return nullopt;
// }
// int best_search(const string &filename,int max_depth=10){
//     ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);
//     VI actions;
//     unordered_set<uint64_t> visit;
//     long count=0;
//     auto res=best_search(initial_state, max_depth, zhash, actions, visit, count);
//     if (res){
//         auto ans=res.value();
//         OUT("update!!!!!", SZ(ans));
//         save_actions(filename, ans);
//         check_answer(filename);
//         return SZ(ans);
//     }
//     return INF;
// }


// void dfs_step(
//     const VI &state, VI &path, int same_action_num, size_t maxdepth, 
//     ZobristHashing<uint64_t> &zhash, unordered_set<uint64_t>& visit, 
//     int &best_mistake, vector<pair<VI, int>> &best_paths
// ){
//     if (path.size() > maxdepth) return;
//     auto hash=zhash.hash(state);
//     if(!path.empty()){
//         if (visit.contains(hash))
//             return;
//         int mis=get_mistakes(state);
//         if(mis<best_mistake){
//             best_paths.clear();
//             best_paths.emplace_back(path, same_action_num);
//             best_mistake=mis;
//         }else if(mis==best_mistake){
//             best_paths.emplace_back(path, same_action_num);
//         }
//     }
//     visit.emplace(hash);
//     REP(a, allowed_action_num*2){
//         int next_same_action_num=1;
//         if(!path.empty()){
//             // 逆操作
//             if(a+allowed_action_num==path.back() || a==path.back()+allowed_action_num)
//                 continue;
//             // グループ順序
//             int prev=(path.back()<allowed_action_num) ? path.back() : (path.back()-allowed_action_num);
//             int act=(a<allowed_action_num) ? a : (a-allowed_action_num);
//             if(to_group_id[prev]==to_group_id[act] && to_order_in_group[prev]>to_order_in_group[act])
//                 continue;
//             // より短い別の動作で置換できる場合
//             if(path.back()==a)
//                 next_same_action_num = same_action_num+1;
//             if(2*next_same_action_num>to_rotate_num[act] 
//                 || (2*next_same_action_num==to_rotate_num[act] && act!=a))
//                 continue;
//         }
//         path.emplace_back(a);
//         dfs_step(do_action(state, a), path, next_same_action_num, maxdepth, zhash, visit, best_mistake, best_paths);
//         path.pop_back();
//     }
// }

// int best_search_step_width(const string &filename, int step_width, int maxdepth){
//     ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);
//     VI actions;

//     unordered_set<uint64_t> visit;
//     auto state=initial_state;
//     int same_action_num=0;

//     while(SZ(actions)<=maxdepth){
//         VI path;
//         int best_mistake=INF;
//         vector<pair<VI, int>> best_paths;

//         dfs_step(state, path, same_action_num, step_width, zhash, visit, best_mistake, best_paths);
//         // dump(SZ(best_paths))
//         dump(best_mistake)
//         dump(SZ(actions))
//         assert(SZ(best_paths)>=1);
//         for(auto&[paths, same]: best_paths){
//             // dump(SZ(paths))
//             for(int a: paths){
//                 actions.emplace_back(a);
//                 state=do_action(state, a);
//             }
//             same_action_num=same;
//             // TODO: 複数にする
//             break;
//         }
//         if(best_mistake<=num_wildcards){
//             OUT("update!!!!!", SZ(actions));
//             save_actions(filename, actions);
//             check_answer(filename);
//             return SZ(actions);
//         }
//     }
//     return INF;
// }


// optional<VI> find_merge_point(const VI &fullpath, int fulltail, const VI &subpath, int subtail, int depth){
//     ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);

//     VVI full_states{initial_state};
//     auto state=initial_state;
//     for(int a: fullpath){
//         state = do_action(state, a);
//         full_states.emplace_back(state);
//     }

//     VVI sub_states{initial_state};
//     state=initial_state;
//     for(int a: subpath){
//         state = do_action(state, a);
//         sub_states.emplace_back(state);
//     }

//     // hash_to_depth_full[0]が末尾状態
//     vector<unordered_map<uint64_t, tuple<int,uint64_t,int>>> hash_to_depth_full(fulltail);
//     #pragma omp parallel for
//     REP(i, fulltail){
//         unordered_map<uint64_t, tuple<int,uint64_t,int>> visit; VI path;
//         dfs(*(full_states.rbegin()+i), path, 0, depth, zhash, visit, 0);
//         hash_to_depth_full[i]=visit;
//         dump(i)
//         dump(hash_to_depth_full[i].size())
//     }

//     // hash_to_depth_sub[0]が末尾状態
//     vector<unordered_map<uint64_t, tuple<int,uint64_t,int>>> hash_to_depth_sub(subtail);
//     #pragma omp parallel for
//     REP(i, subtail){
//         unordered_map<uint64_t, tuple<int,uint64_t,int>> visit; VI path;
//         dfs(*(sub_states.rbegin()+i), path, 0, depth, zhash, visit, 0);
//         hash_to_depth_sub[i]=visit;
//         dump(i)
//         dump(hash_to_depth_sub[i].size())
//     }

//     VI result=subpath;
//     for(int i=0; i<subtail; ++i){
//         for(int j=0; j<fulltail; ++j){
//             for(auto [hash1, value1]: hash_to_depth_sub[i]){
//                 auto it2=hash_to_depth_full[j].find(hash1);
//                 if(it2==hash_to_depth_full[j].end()) continue;

//                 auto act=construct_actions(0, hash1, hash_to_depth_sub[i]);
//                 result.insert(result.end(), act.begin(), act.end());
//                 auto act2=inverse_action(construct_actions(0, it2->first, hash_to_depth_sub[j]));
//                 result.insert(result.end(), act2.begin(), act2.end());
//                 result.insert(result.end(), fullpath.begin()+SZ(fullpath)-1 -j, fullpath.end());
//                 return result;
//             }
//         }
//         result.pop_back();
//     }
//     return nullopt;
// }

// int solve_using_subproblem(const string &fullanswer_path, const string &subanswer_path, int fulltail, int subtail, int depth){
//     auto fullpath=load_actions(fullanswer_path);
//     dump(SZ(fullpath))
//     auto subpath=load_actions(subanswer_path);
//     dump(SZ(subpath))
//     auto result=find_merge_point(fullpath, fulltail, subpath, subtail, depth);
//     if(result){
//         OUT("success merge!!");
//         auto &ans = result.value();
//         int mistake = get_mistakes(simulation(initial_state, ans));
//         assert(mistake<=num_wildcards);
//         if(SZ(ans)<SZ(fullpath)){
//             OUT("saved", SZ(fullpath), "->", SZ(ans));
//             save_actions(fullanswer_path, ans);
//         }
//         return SZ(ans);
//     }else
//         OUT("fail merge!!");
//     return INF;
// }

// VI greedy_improve2(const VI &action, int depth){
//     ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);
//     unordered_map<uint64_t, int> hash_to_idx;
    
//     auto state=initial_state;
//     int idx=0;
//     hash_to_idx[zhash.hash(state)]=idx++;
//     for(int a: action){
//         state = do_action(state, a);
//         hash_to_idx[zhash.hash(state)]=idx++;
//     }

//     VI result;
//     state=initial_state;
//     for(idx=0; idx<SZ(action); ){
//         dump(idx)
//         VI act;
//         int max_idx=-1;
//         int max_improve=-1;
//         // 近傍をチェックして同じ状態があればジャンプする
//         MINPQ<tuple<int, int, int, uint64_t>> pq;
//         auto hash=zhash.hash(state);
//         pq.emplace(0, 0, 0, hash);
//         unordered_map<uint64_t, VI> visit;
//         visit[hash]=VI();
//         int checknum=0;
//         while(!pq.empty()){
//             auto[_, a, same_action_num, hash]=pq.top();pq.pop();
//             if(a+1<allowed_action_num*2)
//                 pq.emplace(_, a+1, same_action_num, hash);
//             auto path=visit[hash];
//             int next_same_action_num=1;
//             if(!path.empty()){
//                 // 逆操作
//                 if(a+allowed_action_num==path.back() || a==path.back()+allowed_action_num)
//                     continue;
//                 // グループ順序
//                 int prev=(path.back()<allowed_action_num) ? path.back() : (path.back()-allowed_action_num);
//                 int act=(a<allowed_action_num) ? a : (a-allowed_action_num);
//                 if(to_group_id[prev]==to_group_id[act] && to_order_in_group[prev]>to_order_in_group[act])
//                     continue;
//                 // より短い別の動作で置換できる場合
//                 if(path.back()==a)
//                     next_same_action_num = same_action_num+1;
//                 if(2*next_same_action_num>to_rotate_num[act] 
//                     || (2*next_same_action_num==to_rotate_num[act] && act!=a))
//                     continue;
//             }
//             path.emplace_back(a);
//             auto next = simulation(state, path);
//             auto next_hash=zhash.hash(next);
//             if(visit.contains(next_hash) && SZ(visit[next_hash])<=SZ(path))
//                 continue;
//             visit[next_hash]=path;

//             if (hash_to_idx.contains(next_hash)){
//                 int tmp_idx = hash_to_idx[next_hash];
//                 int improve = (tmp_idx-idx) - SZ(path);
//                 if (tmp_idx > idx && improve>max_improve){
//                     max_idx=tmp_idx;
//                     max_improve=improve;
//                     act = path;
//                 }
//             }
//             int mistake=get_mistakes(next);
//             if(num_wildcards!=0 && mistake<=num_wildcards){
//                 int improve = (SZ(action)-idx) - SZ(path);
//                 if(max_improve<improve){
//                     max_idx=SZ(action);
//                     max_improve=improve;
//                     act = path;
//                     break;
//                 }
//                 continue;
//             }
//             if (SZ(path)+idx < SZ(action) && SZ(path)<depth)
//                 // pq.emplace(get_penalty(next), 0, next_same_action_num, next_hash);
//                 pq.emplace(mistake, 0, next_same_action_num, next_hash);
//             else{
//                 checknum++;
//                 if(checknum>100)break;
//             }
//         }
//         if(max_idx<0){
//             result.emplace_back(action[idx]);
//             state = do_action(state, action[idx]);
//             idx++;
//         }else{
//             assert(max_idx>idx);
//             if(max_improve>0)
//                 dump(max_improve)
//             idx = max_idx;
//             result.insert(result.end(), act.begin(), act.end());
//             state = simulation(state, act);
//         }
//     }
//     return result;
// }

VI cancel_opposite(const VI& actions){
    VI result;
    MII samegroup;
    int prev_group_id=-1;
    for(int a: actions){
        int act=(a<allowed_action_num) ? a : (a-allowed_action_num);
        if(prev_group_id!=to_group_id[act]){
            for(auto[_a, cnt]: samegroup){
                int loop=abs(cnt)%to_rotate_num[_a];
                int new_a=(cnt<0) ? (_a+allowed_action_num) : _a;
                if(loop > to_rotate_num[_a]/2){
                    new_a=new_a<allowed_action_num? (new_a+allowed_action_num) : (new_a-allowed_action_num);
                    loop=to_rotate_num[_a]-loop;
                }
                REP(_, loop)
                    result.emplace_back(new_a);
            }
            samegroup.clear();
        }
        if(act==a)samegroup[act]++;
        else samegroup[act]--;
        prev_group_id=to_group_id[act];
    }
    for(auto[_a, cnt]: samegroup){
        int loop=abs(cnt)%to_rotate_num[_a];
        int new_a=(cnt<0) ? (_a+allowed_action_num) : _a;
        if(loop > to_rotate_num[_a]/2){
            new_a=new_a<allowed_action_num? (new_a+allowed_action_num) : (new_a-allowed_action_num);
            loop=to_rotate_num[_a]-loop;
        }
        REP(_, loop)
            result.emplace_back(new_a);
    }
    return result;
}

void generate_all(deque<int>& actions, const VI &index, int maxdepth, 
             ZobristHashing<uint64_t> &zhash, unordered_map<uint64_t, set<pair<int, deque<int>>>> &bestaction){
    if(SZ(actions)>maxdepth)return;
    bestaction[zhash.hash(index)].emplace(SZ(actions), actions);
    REP(a, allowed_action_num*2){
        actions.emplace_back(a);
        generate_all(actions, do_action(index, a), maxdepth, zhash, bestaction);
        actions.pop_back();
    }
}

void generate_sorted(deque<int>& path, const VI &index, int same_action_num, int maxdepth, 
             ZobristHashing<uint64_t> &zhash, unordered_map<uint64_t, set<pair<int, deque<int>>>> &bestaction){
    if(SZ(path)>maxdepth)return;
    bestaction[zhash.hash(index)].emplace(SZ(path), path);
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
        path.emplace_back(a);
        generate_sorted(path, do_action(index, a), next_same_action_num, maxdepth, zhash, bestaction);
        path.pop_back();
    }
}

VI kstep_replace(const VI &action, int k, bool sorted){
    map<deque<int>, deque<int>> to_shortest;
    {
        unordered_map<uint64_t, set<pair<int, deque<int>>>> bestaction;
        ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
        VI index(state_length);
        ARANGE(index);
        deque<int> action;
        if(sorted)
            generate_sorted(action, index, 0, k, zhash, bestaction);
        else
            generate_all(action, index, k, zhash, bestaction);
        dump(bestaction.size())

        for(auto& [_, s]: bestaction){
            int minsize=s.begin()->first;
            auto &minpat=s.begin()->second;
            for(auto it=s.rbegin();it!=s.rend() && it->first > minsize;++it)
                to_shortest[it->second]=minpat;
            s.clear();// for memory
        }
        dump(to_shortest.size())
    }
    deque<int> part(action.begin(), action.begin()+min(k, SZ(action)));

    VI result;
    for(int i=k;i<SZ(action); ++i){
        auto it=to_shortest.find(part);
        if(it!=to_shortest.end()){
            OUT("update", SZ(part), "->", SZ(it->second));
            result.insert(result.end(), it->second.begin(), it->second.end());
            part.clear();
            part.insert(part.end(), action.begin()+i, action.end()+min(i+k, SZ(action)));
        }else{
            result.emplace_back(part.front());
            part.pop_front();
            part.emplace_back(action[i]);
        }
    }
    auto it=to_shortest.find(part);
    if(it!=to_shortest.end()){
        OUT("update", SZ(part), "->", SZ(it->second));
        result.insert(result.end(), it->second.begin(), it->second.end());
    }else{
        result.insert(result.end(), part.begin(), part.end());
    }
    return result;
}

optional<VI> rotate_all(const VI& state, int gid, const VI& path, map<int, int> &size_count){
// optional<VI> rotate_all(const VI& state, int gid, const VI& path, int count){
// optional<VI> rotate_all(const VI& state, int gid, const VI& path){
    // if(count<0)
    //     return nullopt;
    if(gid>=SZ(group)){
        if(get_mistakes(state)<=num_wildcards){
            // return cancel_opposite(path);
            return (path);
        }
        return nullopt;
    }
    auto res=rotate_all(state, gid+1, path, size_count);
    // auto res=rotate_all(state, gid+1, path, count);
    // auto res=rotate_all(state, gid+1, path);
    if(res)return res.value();

    auto s=state, p=path;
    int rotate=group_to_rotate_num[gid];
    int loop=(1+rotate)/2;
    int save=size_count[rotate];
    REP(i, loop){
        size_count[rotate]--;
        if(size_count[rotate]<0)
            break;
        for(int a:group[gid]){
            s=do_action(s, a);
            p.emplace_back(a);
        }
        res=rotate_all(s, gid+1, p, size_count);
        // auto res=rotate_all(s, gid+1, p, count-i-1);
        // auto res=rotate_all(s, gid+1, p);
        if(res)return res.value();
    }
    s=state, p=path;
    size_count[rotate]=save;
    REP(i, loop-1){
        size_count[rotate]--;
        if(size_count[rotate]<0)
            break;
        for(int a:group[gid]){
            s=do_action(s, inverse(a));
            p.emplace_back(inverse(a));
        }
        res=rotate_all(s, gid+1, p, size_count);
        // auto res=rotate_all(s, gid+1, p, count-i-1);
        // auto res=rotate_all(s, gid+1, p);
        if(res)return res.value();
    }
    size_count[rotate]=save;
    return nullopt;
}

VI summerize_rotate(const VI& action, int intercept=0){
    // 前処理
    VI index1(state_length), index2(state_length);
    map<int, MII> group_id_to_converter;
    REP(gid, SZ(group)){
        const auto &g=group[gid];
        ARANGE(index1);
        ARANGE(index2);
        for(int a:g)
            index1=do_action(index1, a);
        MII converter;
        REP(a1, allowed_action_num*2){
            auto tmp1=do_action(index1, a1);
            // 対応する操作を求める
            REP(a2, allowed_action_num*2){
                auto tmp2=do_action(index2, a2);
                for(int a:g)
                    tmp2=do_action(tmp2, a);
                if(tmp1==tmp2){
                    converter[a1]=a2;
                    break;
                }
            }
        }
        if(SZ(converter)==allowed_action_num*2)
            group_id_to_converter[gid]=converter;
    }
    // dump(group_id_to_converter)

    VI result;
    VI samegroup(allowed_action_num, 0);
    int prev_group_id=-1;
    // gid, count
    vector<pair<int, int>> mapping;

    auto convert = [&](int i, vector<pair<int, int>> &mapping){
        int new_a=i;
        for(auto&[gid, count]: mapping)REP(_, count)
            new_a=group_id_to_converter[gid][new_a];
        return new_a;
    };
    auto to_base_act=[&](int a){
        return a<allowed_action_num ? a : a-allowed_action_num;
    };

    auto subprocess = [&]() {
        if (group_id_to_converter.contains(prev_group_id)){
            int best_add_rotate=0;
            int best_sumrotate=intercept;
            VI best_new_samegroup(allowed_action_num);
            for(int j: group[prev_group_id]){
                best_new_samegroup[j]=samegroup[j]%to_rotate_num[j];
                if(abs(best_new_samegroup[j])>to_rotate_num[j]/2)
                    best_new_samegroup[j]=best_new_samegroup[j]>0 ? (best_new_samegroup[j]-to_rotate_num[j]) : (best_new_samegroup[j]+to_rotate_num[j]); 
                best_sumrotate+=abs(best_new_samegroup[j]);
            }
            for(int i: group[prev_group_id]){
                VI new_samegroup(allowed_action_num);
                int sumrotate=0;
                for(int j: group[prev_group_id]){
                    new_samegroup[j]=(samegroup[j]-samegroup[i])%to_rotate_num[j];
                    if(abs(new_samegroup[j])>to_rotate_num[j]/2)
                        new_samegroup[j]=new_samegroup[j]>0 ? (new_samegroup[j]-to_rotate_num[j]) : (new_samegroup[j]+to_rotate_num[j]); 
                    sumrotate+=abs(new_samegroup[j]);
                }
                // mappingは正にしておく
                int t=(samegroup[i]%group_to_rotate_num[prev_group_id]+group_to_rotate_num[prev_group_id])%group_to_rotate_num[prev_group_id];
                // if(sumrotate<best_sumrotate || (sumrotate==best_sumrotate && t<best_add_rotate))
                if(sumrotate<best_sumrotate)
                    best_new_samegroup=new_samegroup, best_sumrotate=sumrotate, best_add_rotate=t;
            }
            if(best_add_rotate!=0){
                dump(samegroup)
                dump(best_new_samegroup)
                dump(best_add_rotate)
                mapping.emplace_back(prev_group_id, best_add_rotate);
            }
            samegroup=best_new_samegroup;
        }
        REP(i, allowed_action_num)if(samegroup[i]!=0){
            int new_a=i;
            new_a=(samegroup[i]<0) ? inverse(new_a) : new_a;
            // 
            int loop=abs(samegroup[i]);
            REP(_, loop)
                result.emplace_back(new_a);
        }
    };

    for(int a: action){
        int new_a=convert(a, mapping);
        int act=to_base_act(new_a);
        if(prev_group_id!=to_group_id[act]){
            subprocess();
            fill(ALL(samegroup), 0);
            new_a=convert(a, mapping);
            act=to_base_act(new_a);
        }
        if(new_a<allowed_action_num)samegroup[act]++;
        else samegroup[act]--;
        prev_group_id=to_group_id[act];
    }
    subprocess();

    vector<pair<int, int>> new_mapping;
    int pid=-1, sum=0;
    for(auto&[gid, count]: mapping){
        if(pid!=gid){
            if(pid>=0)
                new_mapping.emplace_back(pid, sum%group_to_rotate_num[pid]);
            pid=gid, sum=count;
        }else
            sum+=count;
    }
    if(pid>=0)
        new_mapping.emplace_back(pid, sum%group_to_rotate_num[pid]);
    mapping=new_mapping;
    map<int, int> group_count;
    for(auto&[gid, count]: mapping){
        group_count[gid]+=count;
        group_count[gid]%=group_to_rotate_num[gid];
    }
    dump(group_count)
    for(auto&[gid, count]: group_count){
        if(count>group_to_rotate_num[gid]/2)
            count=group_to_rotate_num[gid]-count;
    }
    dump(group_count)
    map<int, int> size_count;
    for(auto&[gid, count]: group_count)
        size_count[group_to_rotate_num[gid]]+=count;

    // TODO: ベスト挿入箇所を探索する
    dump(size_count)
    // auto add_rotate=rotate_all(simulation(initial_state, result), 0, VI());
    // auto add_rotate=rotate_all(simulation(initial_state, result), 0, VI(), need_rotate);
    // 遅い場合がある
    auto add_rotate=rotate_all(simulation(initial_state, result), 0, VI(), size_count);
    while(!add_rotate){
        for(auto &[size, count]: size_count)
            count++;
        dump(size_count)
        add_rotate=rotate_all(simulation(initial_state, result), 0, VI(), size_count);
    }
    dump(add_rotate.value().size())
    result.insert(result.end(), add_rotate.value().begin(), add_rotate.value().end());

    // NOTE: 回転をそのまま使うと系列が長くなる
    // // dump(solution_state)
    // // dump(mapping)
    // RITR(it, mapping){
    //     auto&[gid, count] = *it;
    //     // countは正
    //     int loop=count>group_to_rotate_num[gid]/2 ? (count-group_to_rotate_num[gid]) : count;
    //     bool inv = (loop<0);
    //     loop=abs(loop);
    //     for(int a: group[gid]){
    //         int act = inv ? inverse(a) : a;
    //         REP(_, loop)
    //             result.emplace_back(act);
    //     }
    // }
    // // dump(simulation(initial_state, result))
    return result;
}


// 解の圧縮
int compression(const string &filename, int depth, int search_step=INF, int random_prune=0){
    auto actions=load_actions(filename);
    dump(SZ(actions))

    auto result = actions;
    // result = wildcard_finish(result);
    // result = cancel_opposite(result);
    // result = kstep_replace(result, depth, true);
    // result = kstep_replace(result, depth, false);
    result=summerize_rotate(result, 0);
    // result=summerize_rotate(result, 1);
    // result=summerize_rotate(result, -1);
    // result = dual_greedy_improve(result, min(depth, SZ(actions)), search_step, random_prune);
    // result = dual_greedy_improve_low_memory(result, min(depth, SZ(actions)), search_step);
    // result = greedy_improve(result, depth);
    result = same_state_skip(result);
    result = loop_compress(result);

    int mistake = get_mistakes(simulation(initial_state, result));
    assert(mistake<=num_wildcards);
    if(SZ(result)<SZ(actions)){
        OUT("saved", SZ(actions), "->", SZ(result));
        save_actions(filename, result);
    }
    return SZ(result);
}

// struct WreathSolver{
//     int a;// 右 red
//     int b;// 左 blue
//     int c;
//     int Cindex1, Cindex2;
//     int l_stepsize;
//     int r_stepsize;
//     VI matched;
//     VI leftindex;
//     VI rightindex;
//     WreathSolver(){
//         a=label_mapping["A"];// 右 red
//         b=label_mapping["B"];// 左 blue
//         c=label_mapping["C"];
//         Cindex1=-1, Cindex2=-1;
//         // A B A C A B B B A C
//         // C A C A A A B B B B
//         REP(i, state_length)if(solution_state[i]==c){
//             if(Cindex1<0) Cindex1=i;
//             else if(Cindex2<0) Cindex2=i;
//             else assert(false);
//         }
//         l_stepsize=0;
//         r_stepsize=0;
//         auto state=solution_state;
//         REP(act, allowed_action_num)
//             REP(i, to_rotate_num[act]){
//                 state=do_action(state, 0);
//                 if(state[Cindex1]==c||state[Cindex2]==c){
//                     if(act==0) l_stepsize=i+1;
//                     else r_stepsize=i+1;
//                     break;
//                 }
//             }
//         dump(l_stepsize)
//         dump(r_stepsize)

//         REP(act, allowed_action_num){
//             VI index(state_length);
//             ARANGE(index);
//             if(act==0)leftindex.emplace_back(index[Cindex1]);
//             else rightindex.emplace_back(index[Cindex1]);
//             REP(i, to_rotate_num[act]-1){
//                 index=do_action(index, act);
//                 if(act==0)leftindex.emplace_back(index[Cindex1]);
//                 else rightindex.emplace_back(index[Cindex1]);
//             }
//         }
//         dump(leftindex)
//         dump(rightindex)
//         matching(initial_state);
//     }
//     void matching(const VI& state){
//         matched.assign(state_length, -1);
//         REP(act, allowed_action_num){
//             auto &index= (act==0) ? leftindex : rightindex;
//             int stepsize = (act==0) ? l_stepsize : r_stepsize;
//             int color = (act==0) ? b : a;
//             REP(i, SZ(index)){
//                 int I=index[i];
//                 if(matched[I]<0){
//                     int J=index[(i+stepsize)%SZ(index)];
//                     if(matched[J]<0 && state[I]==color && state[J]==color){
//                         matched[J]=I, matched[I]=J;
//                         continue;
//                     }
//                     J=index[(i-stepsize+SZ(index))%SZ(index)];
//                     if(matched[J]<0 && state[I]==color && state[J]==color){
//                         matched[J]=I, matched[I]=J;
//                         continue;
//                     }
//                 }
//             }
//         }
//         dump(matched);
//     }

//     tuple<int, bool, bool> get_loop_num(const VI& istate, int act){
//         int color = act==0 ? b : a;
//         auto state=istate;
//         int num=to_rotate_num[act]-1;
//         if(get_mistakes(state)<=num_wildcards)
//             return {0, true, false};
//         else if(state[Cindex1]==color && color==state[Cindex2])
//             return {0, false, false};
//         REP(loop, num){
//             state=do_action(state, act);
//             bool inv=2*(loop+1) > to_rotate_num[act];
//             if(get_mistakes(state)<=num_wildcards)
//                 return {inv ? (to_rotate_num[act]-loop-1) : (loop+1), true, inv};
//             else if(state[Cindex1]==color && color==state[Cindex2])
//                 return {inv ? (to_rotate_num[act]-loop-1) : (loop+1), false, inv};
//         }
//         return {-1, false, false};
//     }
//     optional<VI> solve(int current_best_size){
//         // lを繰り返して、2つのaを見つける → rを回転する
//         // rを繰り返して、2つのbを見つける → lを回転する
//         VI actions;
//         auto state=initial_state;
//         while(1){
//             auto [left_num, lfin, linv]=get_loop_num(state, 0);
//             auto [right_num, rfin, rinv]=get_loop_num(state, 1);
            
//             if(lfin && rfin){
//                 if(left_num<right_num)
//                     REP(_, left_num)actions.emplace_back(linv ? 2 : 0);
//                 else
//                     REP(_, right_num)actions.emplace_back(rinv ? 3 : 1);
//                 break;
//             }else if(lfin){
//                 REP(_, left_num)actions.emplace_back(linv ? 2 : 0);
//                 break;
//             }else if(rfin){
//                 REP(_, right_num)actions.emplace_back(rinv ? 3 : 1);
//                 break;
//             }else if(left_num>0 && right_num>0){
//                 if(left_num<right_num)
//                     REP(_, left_num){
//                         int act=linv ? 2 : 0;
//                         actions.emplace_back(act);
//                         state=do_action(state, act);
//                     }
//                 else
//                     REP(_, right_num){
//                         int act=rinv ? 3 : 1;
//                         actions.emplace_back(act);
//                         state=do_action(state, act);
//                     }
//             }else if(left_num>0){
//                 REP(_, left_num){
//                     int act=linv ? 2 : 0;
//                     actions.emplace_back(act);
//                     state=do_action(state, act);
//                 }
//             }else if(right_num>0){
//                 REP(_, right_num){
//                     int act=rinv ? 3 : 1;
//                     actions.emplace_back(act);
//                     state=do_action(state, act);
//                 }
//             }else{
//                 dump(action_decode(actions))
//                 dump(state)

//                 // IDAstar idastar;
//                 // bool dual=false;
//                 // auto res = dual ?
//                 //     idastar.ida_star_bidirect(state, solution_state, current_best_size, num_wildcards)
//                 //     : idastar.ida_star(state, solution_state, current_best_size, num_wildcards);
//                 auto res=search(state, solution_state, current_best_size, false, num_wildcards);
//                 if(res){
//                     auto& add = res.value();
//                     actions.insert(actions.end(), add.begin(), add.end());
//                     return actions;
//                 }
//                 return nullopt;
//             }
//             // dump(state)
//         }
//         return actions;
//     }
// };

// int run_WreathSolver(const string &filename){
//     assert(file_exists(filename));
//     auto current_best=load_actions(filename);
//     dump(SZ(current_best))
    
//     WreathSolver solver;
//     auto res=solver.solve(SZ(current_best));
//     if(res){
//         if(SZ(res.value())<SZ(current_best)){
//             OUT("saved", SZ(current_best), "->", SZ(res.value()));
//             save_actions(filename, res.value());
//         }
//         return SZ(res.value());
//     }
//     return INF;
// }

// optional<VI> rotate_all(const VI& state, int gid, const VI& path){
//     if(gid>=SZ(group)){
//         if(get_mistakes(state)<=num_wildcards)
//             return path;
//         return nullopt;
//     }
//     auto res=rotate_all(state, gid+1, path);
//     if(res)return res.value();

//     auto s=state, p=path;
//     int rotate=group_to_rotate_num[gid];
//     int loop=(1+rotate)/2;
//     int save=size_count[rotate];
//     REP(i, loop){
//         size_count[rotate]--;
//         if(size_count[rotate]<0)
//             break;
//         for(int a:group[gid]){
//             s=do_action(s, a);
//             p.emplace_back(a);
//         }
//         auto res=rotate_all(s, gid+1, p);
//         if(res)return res.value();
//     }
//     s=state, p=path;
//     size_count[rotate]=save;
//     REP(i, loop-1){
//         size_count[rotate]--;
//         if(size_count[rotate]<0)
//             break;
//         for(int a:group[gid]){
//             s=do_action(s, inverse(a));
//             p.emplace_back(inverse(a));
//         }
//         auto res=rotate_all(s, gid+1, p);
//         if(res)return res.value();
//     }
//     size_count[rotate]=save;
//     return nullopt;
// }


const string DATA_DIR = "./data/";
map<string, int> TARGET{
       {"cube_2/2/2", 5},
       {"cube_3/3/3", 4},
       {"cube_4/4/4", 4},
       {"cube_5/5/5", 3},
       {"cube_6/6/6", 3},
       {"cube_7/7/7", 3},
       {"cube_8/8/8", 3},
       {"cube_9/9/9", 3},
       {"cube_10/10/10", 3},
       {"cube_19/19/19", 2},
       {"cube_33/33/33", 1},
       {"wreath_6/6", 12},
       {"wreath_7/7", 12},
       {"wreath_12/12", 12},
       {"wreath_21/21", 12},
       {"wreath_33/33", 12},
       {"wreath_100/100", 9},
       {"globe_1/8", 4},
       {"globe_1/16", 3},
       {"globe_2/6", 4},
       {"globe_3/4", 4},
       {"globe_6/4", 4},
       {"globe_6/8", 3},
       {"globe_6/10", 3},
       {"globe_3/33", 2},
       {"globe_8/25", 2}
};

int main() {
    ChronoTimer timer;
    int case_num = 398;
    double sum_score = 0.0;
    double sum_log_score = 0.0;
    int64_t max_time = 0;
    // for(int i: VI{200000, 201000, 202000, 204000, 205000, 206000, 207000, 208000, 209000, 240000, 241000, 242000, 243000, 244000, 255000, 256000, 283000}){
    // for(int i: VI{200, 201, 202, 204, 205, 206, 207, 208, 209, 240, 241, 242, 243, 244, 255, 256, 283}){
    // for(int i: VI{255000, 256000, 283000}){
    // for(int i: VI{240000, 241000, 242000, 243000, 244000, 255000, 256000, 283000}){
    // for(int i: VI{283000}){
    // for(int i: VI{259}){
    REP(i, case_num){
    // RREP(i, case_num){
    // FOR(i,  150, case_num){
    // FOR(i,  130, 150){
    // FOR(i, 284, case_num){
    // FOR(i, 282, 282+1){
    // FOR(i, 348, case_num){
    // FOR(i, 353, case_num){
    // FOR(i, 337, case_num){
    // FOR(i, 358, case_num){
    // FOR(i, 332, case_num){
        timer.start();
        dump(SEED)
        rand_engine.seed(SEED);
        string input_filename = to_string(i) + ".txt";
        string file_path = DATA_DIR + input_filename;
        ifstream ifs(file_path);
        assert(!ifs.fail());
        data_load(ifs);
        if(!TARGET.contains(puzzle_type))
            continue;
        OUT("id", i);
        // if(num_wildcards==0)continue;
        double score=0;

        string output_filename="output/"+to_string(i)+".txt";
        // score=ida_solve(output_filename, false);
        // score=ida_solve(output_filename, true);

        // score = annealing(output_filename, timer, 5000, 100);
        // score = annealing(output_filename, timer, 100, 100);
        // score = search(output_filename, false, 1);
        // score = search(output_filename, true, 1);
        // score=run_WreathSolver(output_filename);

        // score=solve_using_subproblem(output_filename, "subanswer/"+to_string(i*1000)+".txt", 50, 10, 4);
        // best_search_step_width(output_filename, 4, 1000);
 
        // score=compression(output_filename, TARGET[puzzle_type]);
        score=compression(output_filename, TARGET[puzzle_type], 100);
        // score=compression(output_filename, TARGET[puzzle_type]+1, 100, 25);
        // score=compression(output_filename, TARGET[puzzle_type]+2, 100, 60);

        score=check_answer(output_filename);
        timer.end();
        if(DEBUG) {
            auto time = timer.time();
            sum_score += score;
            sum_log_score += log1p(score);
            max_time = max(max_time, time);
            OUT("--------------------");
            OUT("case_num: ", i);
            OUT("puzzle_type: ", puzzle_type);
            OUT("allowed_action_num: ", allowed_action_num);
            OUT("num_wildcards: ", num_wildcards);
            OUT("score: ", score);
            OUT("time: ", time);
            OUT("sum_score: ", sum_score);
            OUT("sum_log_score: ", sum_log_score);
            OUT("max_time: ", max_time);
            OUT("--------------------");
        }
    }
    return 0;
}
