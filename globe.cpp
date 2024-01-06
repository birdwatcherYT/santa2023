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

template<class T> vector<int> argsort(const vector<T> &vec, bool asc=true){
    vector<int> index(vec.size()); iota(index.begin(), index.end(), 0);
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
        // REP(i, size){
        REP(i, SZ(array)){
            // (i, e) → i*nunique + e
            auto &e=array[i];
            value ^= h[i*nunique + e];
        }
        return value;
    }
    T hash(const VVI &mat) {
        T value = 0;
        int num=0;
        REP(k, SZ(mat))REP(i, SZ(mat[k])){
            auto &e=mat[k][i];
            value ^= h[num*nunique + e];
            num++;
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
map<string,int> allowed_moves_name_to_int;
// VVI allowed_moves;
vector<unordered_map<int,int>> allowed_moves;
int num_wildcards;
// 逆操作
// VVI allowed_moves_inverse;
vector<unordered_map<int,int>> allowed_moves_inverse;
// vector<PII> same_color_index;

#define get_action(id) ((id)<allowed_action_num ? allowed_moves[id] : allowed_moves_inverse[(id)-allowed_action_num])
#define get_action_name(id) ((id)<allowed_action_num ? allowed_moves_name[id] : ("-"+allowed_moves_name[(id)-allowed_action_num]))

VI do_action(const VI& state, const VI &action){
    auto s=state;
    REP(i, SZ(state))
        s[i]=state[action[i]];
    return s;
}

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

void data_load(istream &is){
    // データ読み込み ----------
    is  >> puzzle_type
        >> state_length;
    VS initial_str(state_length);
    VS solution_str(state_length);
    is>>initial_str>>solution_str;
    
    initial_state.assign(state_length, 0);
    solution_state.assign(state_length, 0);
    label_mapping.clear();
    REP(i, state_length){
        auto &label=solution_str[i];
        if(!label_mapping.contains(label))
            label_mapping[label]=SZ(label_mapping);
        solution_state[i]=label_mapping[label];
    }
    REP(i, state_length){
        auto &label=initial_str[i];
        assert(label_mapping.contains(label));
        initial_state[i]=label_mapping[label];
    }
    is >> allowed_action_num;
    allowed_moves_name.assign(allowed_action_num, "");
    // allowed_moves.assign(allowed_action_num, VI(state_length));
    allowed_moves.assign(allowed_action_num, unordered_map<int,int>());
    REP(i, allowed_action_num){
        is  >> allowed_moves_name[i];
            // >> allowed_moves[i];
        allowed_moves_name_to_int[allowed_moves_name[i]]=i;
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
    
    auto tmp=split_str(puzzle_type, '_', true);
    kind=tmp[0];
    tmp=split_str(tmp[1], '/', true);
    X = stoi(tmp[0]);
    Y = stoi(tmp[1]);

    dump(puzzle_type)
    dump(num_wildcards)
    dump(allowed_action_num)
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


// 保存した解のチェック
int check_answer(const string &filename){
    if(!file_exists(filename))
        return INF;
    auto actions=load_actions(filename);
    auto result = simulation(initial_state, actions);
    int mistake = get_mistakes(result);
    assert(mistake<=num_wildcards);
    return SZ(actions);
}
/////////////////////////////////////////////////////////////////////////////


// string_upper = "_".join(["."] + x[:2 * n] + x[:2 * n] + ["."]);
int string_upper_find(VI& x, VI& x0, bool rev=false){
    assert(SZ(x)%4==0);
    int n = SZ(x)/4, len=SZ(x0);
    for(int i=0; i<=4*n-len; ++i){
        int j;
        for(j=0; j<len; ++j)
            if(x[(i+j)%(2*n)]!=x0[rev ? len-1-j : j])break;
        if(j==len)return i;
    }
    return -1;
}

// string_lower = "_".join(["."] + x[2 * n:] + x[2 * n:] + ["."]);
int string_lower_find(VI& x, VI& x0, bool rev=false){
    assert(SZ(x)%4==0);
    int n = SZ(x)/4, len=SZ(x0);
    for(int i=0; i<=4*n-len; ++i){
        int j;
        for(j=0; j<len; ++j)
            if(x[2*n+(i+j)%(2*n)]!=x0[rev ? len-1-j : j])break;
        if(j==len)return i;
    }
    return -1;
}

int heuristic_0(VI &x, VVI &done_list){
    int res = 4;
    int n = SZ(x)/4;
    assert(done_list.size()==4);

    int count_upper = 0;
    int count_lower = 0;
    if (string_upper_find(x, done_list[0]) >= 0){
        res -= 1;
        count_upper += 1;
    }else if (string_lower_find(x, done_list[0], true) >= 0){
        res -= 1;
        count_lower += 1;
    }
    if (string_upper_find(x, done_list[1]) >= 0){
        res -= 1;
        count_upper += 1;
    }else if (string_lower_find(x, done_list[1], true) >= 0){
        res -= 1;
        count_lower += 1;
    }
    if (string_lower_find(x, done_list[2]) >= 0){
        res -= 1;
        count_lower += 1;
    }else if (string_upper_find(x, done_list[2], true) >= 0){
        res -= 1;
        count_upper += 1;
    }
    if (string_lower_find(x, done_list[3]) >= 0){
        res -= 1;
        count_lower += 1;
    }else if (string_upper_find(x, done_list[3], true) >= 0){
        res -= 1;
        count_upper += 1;
    }
    if (max(count_lower, count_upper) >= 3 && SZ(done_list[3]) >= max(n / 3, 2))
        res += 10;
    return res;
}

int heuristic(VI &x, VI &y, VVI &done_list, int base_index = 0, optional<int> add=nullopt){
    int h0 = heuristic_0(x, done_list);
    int n = SZ(x) / 4;
    if(!add)return h0;
    if (h0 > 0) return h0 * 1000;
    // # find index for base
    auto &base_list = done_list[base_index];
    auto &x_joint = base_list;
    auto x_joint_add = base_list; x_joint_add.emplace_back(add.value());

    if (base_index ==0 || base_index==1){
        if (string_upper_find(x, x_joint_add)>=0 || string_lower_find(x, x_joint_add, true)>=0)
            return 0;
    }else{
        if (string_lower_find(x, x_joint_add)>=0 || string_upper_find(x, x_joint_add, true)>=0)
            return 0;
    }

    VI x_;
    int start_ind;
    if (base_index ==0 || base_index==1){
        int s_up = string_upper_find(x, x_joint);
        if (s_up >= 0){
            // start_ind = string_upper[:s_up].count("_") - 1;
            x_ = x;
            start_ind = s_up - 1;
        }else{
            // x_ = list(reversed(x[2 * n:])) + list(reversed(x[:2 * n]));
            x_.insert(x_.end(), x.rbegin(), x.rend());
            int s_up = string_upper_find(x_, x_joint);
            assert (s_up >= 0);
            start_ind = s_up - 1;
        }
    }else{
        int s_low = string_lower_find(x, x_joint);
        if (s_low >= 0){
            start_ind = s_low - 1;
            // x_ = x[2 * n:] + x[:2 * n];
            x_.insert(x_.end(), x.begin()+2*n, x.end());
            x_.insert(x_.end(), x.begin(), x.begin()+2*n);
        }else{
            // x_ = list(reversed(x[:2 * n])) + list(reversed(x[2 * n:]));
            RREP(i,2*n)x_.emplace_back(x[i]);
            RFOR(i,2*n, SZ(x))x_.emplace_back(x[i]);
            s_low = string_upper_find(x_, x_joint);
            assert (s_low >= 0);
            start_ind = s_low - 1;
        }
    }
    int res = 10000000;
    int last_ind = start_ind + SZ(base_list) - 1;
    REP(i_add, 4 * n){
        if (x_[i_add] == add && (i_add < start_ind || last_ind < i_add)){
            int a = min((last_ind + 1) % n, (((-(last_ind + 1)) % n) + n)%n);
            int b = min(i_add % n, (((-i_add) % n)+n)%n);
            res = min(res, a + b + 1);
        }
    }
    return res;
}

pair<VI, VS> add_one(
    VI &initial_state, 
    VI &goal_state, 
    map<string, VI> &allowed_moves_mod,
    VVI& done_list, 
    int base_index = 0, 
    optional<int> add=nullopt,
    int center = -1
){
    // int n = SZ(initial_state) / 4;
    assert (SZ(initial_state) % 4 == 0);
    assert (SZ(initial_state) == SZ(goal_state));
    ZobristHashing<uint64_t> zhash(1+MAX(initial_state), SZ(initial_state), rand_engine);

    MINPQ<tuple<int, VI, VS>> open_set;

    open_set.emplace(0, initial_state, VS());
    unordered_set<uint64_t> closed_set;
    // set<VI> closed_set;

    while (!open_set.empty()){
        auto [_, current_state, path] = open_set.top(); open_set.pop();
        int h = heuristic(current_state, goal_state, done_list, base_index, add);
        if (h == 0)
            // # print(current_state, path)
            return {current_state, path};
        // # print(h, current_state)

        if (current_state == goal_state)
            return {current_state, path};

        closed_set.emplace(zhash.hash(current_state));
        // closed_set.emplace((current_state));

        VS action_list;
        if (center != -1){
            // action_list = [["r0"], ["-r0"], ["r1"], ["-r1"], [f"f{center}"]];
            action_list.emplace_back("r0");
            action_list.emplace_back("-r0");
            action_list.emplace_back("r1");
            action_list.emplace_back("-r1");
            action_list.emplace_back("f"+to_string(center));
        }else{
            // action_list = [[k] for k in allowed_moves.keys()];
            for(auto &[k,v]: allowed_moves_mod)
                action_list.emplace_back(k);
        }
        for (auto &action :action_list){
            auto new_state = do_action(current_state, allowed_moves_mod[action]);
            auto new_hash = zhash.hash(new_state);
            if (!closed_set.contains(new_hash)){
            // if (!closed_set.contains(new_state)){
                int h_new = heuristic(new_state, goal_state, done_list, base_index, add);
                int priority = SZ(path) + 1 + h_new;
                path.emplace_back(action);
                open_set.emplace(priority, new_state, path);
                path.pop_back();
            }
        }
    }
    return pair<VI,VS>();
}

optional<pair<VVI,VS>> solve_greed(
    VVI &initial_state, VVI &goal_state, 
    optional<VI> length_list_or_null = nullopt,
    int r_0 = 0, int r_1 = 0
){
    assert (SZ(initial_state) ==2 && SZ(goal_state) == 2);
    // assert (list(sorted(initial_state[0] + initial_state[1])) == list(sorted(goal_state[0] + goal_state[1])));
    if(DEBUG){
        VI init=initial_state[0];
        init.insert(init.end(), initial_state[1].begin(), initial_state[1].end());
        SORT(init);

        VI goal=goal_state[0];
        goal.insert(goal.end(), goal_state[1].begin(), goal_state[1].end());
        SORT(goal);
        
        assert(init==goal);
    }
    VI length_list;
    if (!length_list_or_null){
        // length_list = [1] * (max(initial_state[0] + initial_state[1]) + 1);
        length_list.assign(max(MAX(initial_state[0]), MAX(initial_state[1])) + 1, 1);
    }else
        length_list=length_list_or_null.value();
    // assert sum([length_list[p] for p in initial_state[0]]) == sum([length_list[p] for p in initial_state[1]])
    if(DEBUG){
        int s0=0;
        for(int p : initial_state[0])s0+=length_list[p];
        int s1=0;
        for(int p : initial_state[1])s1+=length_list[p];
        assert(s0==s1);
    }
    // assert sum([length_list[p] for p in goal_state[0]]) == sum([length_list[p] for p in goal_state[1]])
    if(DEBUG){
        int s0=0;
        for(int p : goal_state[0])s0+=length_list[p];
        int s1=0;
        for(int p : goal_state[1])s1+=length_list[p];
        assert(s0==s1);
    }
    // assert sum([length_list[p] for p in initial_state[0]]) == sum([length_list[p] for p in goal_state[0]])
    if(DEBUG){
        int s0=0;
        for(int p : initial_state[0])s0+=length_list[p];
        int s1=0;
        for(int p : goal_state[0])s1+=length_list[p];
        assert(s0==s1);
    }

    int n = 0;
    for(int p: initial_state[0])
        n+=length_list[p];
    assert(n%2==0);
    n/=2;

    MINPQ<tuple<int,VVI,VS>> open_set;
    int c_0 = length_list[initial_state[0].back()];
    int c_1 = length_list[initial_state[1].back()];
    // # right-right
    VS action;
    REP(i,r_0)action.emplace_back("r0");
    REP(i,r_1)action.emplace_back("r1");
    open_set.emplace(r_0 + r_1, initial_state, action);

    // # left-right
    VVI new_state(2, VI());
    new_state[0].emplace_back(initial_state[0].back());
    new_state[0].insert(new_state[0].end(), initial_state[0].begin(), initial_state[0].begin()+SZ(initial_state[0])-1);
    new_state[1]=initial_state[1];
    action.clear();
    if (c_0 > r_0){
        REP(i,c_0 - r_0)action.emplace_back("-r0");
        REP(i,r_1)action.emplace_back("r1");
        open_set.emplace((c_0 - r_0) + r_1, new_state, action);
    }else{
        REP(i,r_0 - c_0)action.emplace_back("r0");
        REP(i,r_1)action.emplace_back("r1");
        open_set.emplace((r_0 - c_0) + r_1, new_state, action);
    }

    // # right-left
    new_state[0]=initial_state[0];
    new_state[1].clear();
    new_state[1].emplace_back(initial_state[1].back());
    new_state[1].insert(new_state[1].end(), initial_state[1].begin(), initial_state[1].begin()+SZ(initial_state[1])-1);
    action.clear();
    if (c_1 > r_1){
        REP(i,c_1 - r_1)action.emplace_back("-r1");
        REP(i,r_0)action.emplace_back("r0");
        open_set.emplace((c_1 - r_1) + r_0, new_state, action);
    }else{
        REP(i,r_1 - c_1)action.emplace_back("r1");
        REP(i,r_0)action.emplace_back("r0");
        open_set.emplace((r_1 - c_1) + r_0, new_state, action);
    }
    // # left-left
    new_state[0].clear();
    new_state[0].emplace_back(initial_state[0].back());
    new_state[0].insert(new_state[0].end(), initial_state[0].begin(), initial_state[0].begin()+SZ(initial_state[0])-1);
    new_state[1].clear();
    new_state[1].emplace_back(initial_state[1].back());
    new_state[1].insert(new_state[1].end(), initial_state[1].begin(), initial_state[1].begin()+SZ(initial_state[1])-1);
    action.clear();
    if (c_0 > r_0){
        REP(i,c_0 - r_0)action.emplace_back("-r0");
        if (c_1 > r_1){
            REP(i,c_1 - r_1)action.emplace_back("-r1");
            open_set.emplace(abs(c_0 - r_0) + abs(c_1 - r_1), new_state, action);
        }else{
            REP(i,r_1 - c_1)action.emplace_back("r1");
            open_set.emplace(abs(c_0 - r_0) + abs(c_1 - r_1), new_state, action);
        }
    }else{
        REP(i,r_0 - c_0)action.emplace_back("r0");
        if (c_1 > r_1){
            REP(i,c_1 - r_1)action.emplace_back("-r1");
            open_set.emplace(abs(c_0 - r_0) + abs(c_1 - r_1), new_state, action);
        }else{
            REP(i,r_1 - c_1)action.emplace_back("r1");
            open_set.emplace(abs(c_0 - r_0) + abs(c_1 - r_1), new_state, action);
        }
    }
    
    ZobristHashing<uint64_t> zhash(1+max(MAX(initial_state[0]), MAX(initial_state[1])), SZ(initial_state[0])+SZ(initial_state[1]), rand_engine);
    unordered_set<uint64_t> closed_set;
    // set<VVI> closed_set;

    auto heuristic=[](VVI&s, VVI&g){
        return 0;
    };

    while (!open_set.empty()){
        auto [_, current_state, path] = open_set.top(); open_set.pop();

        if (current_state == goal_state)
            return pair<VVI, VS>{current_state, path};

        closed_set.emplace(zhash.hash(current_state));
        // closed_set.emplace((current_state));
        // # r0
        new_state = current_state;
        // TODO: actionをif内に入れてnew_stateをcurrentに変える
        action.clear();
        REP(_, length_list[new_state[0][0]]) action.emplace_back("r0");
        // new_state[0] = new_state[0][1:] + [new_state[0][0]];
        new_state[0].clear();
        new_state[0].insert(new_state[0].end(), current_state[0].begin()+1, current_state[0].end());
        new_state[0].emplace_back(current_state[0][0]);
        if (!closed_set.contains(zhash.hash(new_state))){
        // if (!closed_set.contains((new_state))){
            int priority = SZ(path) + SZ(action) + heuristic(new_state, goal_state);
            auto p=path;
            p.insert(p.end(), action.begin(), action.end());
            open_set.emplace(priority, new_state, p);
        }

        // # -r0
        new_state = current_state;
        action.clear();
        REP(_, length_list[new_state[0].back()]) action.emplace_back("-r0");
        // new_state[0] = [new_state[0][-1]] + new_state[0][:-1]
        new_state[0].clear();
        new_state[0].emplace_back(current_state[0].back());
        new_state[0].insert(new_state[0].end(), current_state[0].begin(), current_state[0].begin()+SZ(current_state[0])-1);
        if (!closed_set.contains(zhash.hash(new_state))){
        // if (!closed_set.contains((new_state))){
            int priority = SZ(path) + SZ(action) + heuristic(new_state, goal_state);
            auto p=path; p.insert(p.end(), action.begin(), action.end());
            open_set.emplace(priority, new_state, p);
        }
        // # r1
        new_state = current_state;
        action.clear();
        REP(_, length_list[new_state[1][0]]) action.emplace_back("r1");
        // new_state[1] = new_state[1][1:] + [new_state[1][0]]
        new_state[1].clear();
        new_state[1].insert(new_state[1].end(), current_state[1].begin()+1, current_state[1].end());
        new_state[1].emplace_back(current_state[1][0]);
        if (!closed_set.contains(zhash.hash(new_state))){
        // if (!closed_set.contains((new_state))){
            int priority = SZ(path) + SZ(action) + heuristic(new_state, goal_state);
            auto p=path; p.insert(p.end(), action.begin(), action.end());
            open_set.emplace(priority, new_state, p);
        }
        // # -r1
        new_state = current_state;
        action.clear();
        REP(_, length_list[new_state[1].back()]) action.emplace_back("-r1");
        // new_state[1] = [new_state[1][-1]] + new_state[1][:-1]
        new_state[1].clear();
        new_state[1].emplace_back(current_state[1].back());
        new_state[1].insert(new_state[1].end(), current_state[1].begin(), current_state[1].begin()+SZ(current_state[1])-1);
        if (!closed_set.contains(zhash.hash(new_state))){
        // if (!closed_set.contains((new_state))){
            int priority = SZ(path) + SZ(action) + heuristic(new_state, goal_state);
            auto p=path; p.insert(p.end(), action.begin(), action.end());
            open_set.emplace(priority, new_state, p);
        }
        // # fn
        int up_total = 0;
        int i_up = -1;
        REP(i, SZ(current_state[0])){
            int a=current_state[0][i];
            up_total += length_list[a];
            if (up_total == n)
                i_up = i + 1;
        }
        int down_total = 0;
        int i_down = -1;
        REP(i, SZ(current_state[1])){
            int a=current_state[1][i];
            down_total += length_list[a];
            if (down_total == n)
                i_down = i + 1;
        }
        if (i_up > 0 && i_down > 0){
            // new_state = [
            //     current_state[0][:i_up] + list(reversed(current_state[1][i_down:])),
            //     current_state[1][:i_down] + list(reversed(current_state[0][i_up:]))
            // ];
            new_state[0].clear();
            new_state[0].insert(new_state[0].end(), current_state[0].begin(), current_state[0].begin()+i_up);
            for(int i=SZ(current_state[1])-1; i>=i_down; --i)
                new_state[0].emplace_back(current_state[1][i]);
            new_state[1].clear();
            new_state[1].insert(new_state[1].end(), current_state[1].begin(), current_state[1].begin()+i_down);
            for(int i=SZ(current_state[0])-1; i>=i_up; --i)
                new_state[1].emplace_back(current_state[0][i]);

            action.clear();
            action.emplace_back("f"+to_string(n));
            if (!closed_set.contains(zhash.hash(new_state))){
            // if (!closed_set.contains((new_state))){
                int priority = SZ(path) + SZ(action) + heuristic(new_state, goal_state);
                auto p=path; p.insert(p.end(), action.begin(), action.end());
                open_set.emplace(priority, new_state, p);
            }
        }
    }
    return nullopt;
}


optional<pair<VVI,VS>> solve_last(VI &state, VI &goal_state, VVI &done_list){
    auto &x = state;
    int n = SZ(x) / 4;
    auto &x0 = done_list[0];
    auto &x1 = done_list[1];
    auto &x2 = done_list[2];
    auto &x3 = done_list[3];
    int x4 = goal_state[n - 1];
    int x5 = goal_state[2 * n - 1];
    int x6 = goal_state[3 * n - 1];
    int x7 = goal_state[4 * n - 2];
    // # print(x)
    // # print(done_list)
    // # print(x4, x5, x6, x7)

    VI res_up(2 * n, -1);
    VI res_down(2 * n, -1);
    int s_up = string_upper_find(x, x0);
    if (s_up >= 0){
        int st = s_up;
        FOR(i, st, st + SZ(done_list[0]))
            res_up[i % (2 * n)] = 0;
    }else{
        int s_down = string_lower_find(x, x0, true);
        int st = s_down;
        FOR(i, st, st + SZ(done_list[0]))
            res_down[i % (2 * n)] = 0;
    }
    s_up = string_upper_find(x, x1);
    if (s_up >= 0){
        int st = s_up;
        FOR(i, st, st + SZ(done_list[1]))
            res_up[i % (2 * n)] = 1;
    }else{
        int s_down = string_lower_find(x, x1, true);
        int st = s_down;
        FOR(i, st, st + SZ(done_list[1]))
            res_down[i % (2 * n)] = 1;
    }
    s_up = string_upper_find(x, x2, true);
    if (s_up >= 0){
        int st = s_up;
        FOR(i, st, st + SZ(done_list[2]))
            res_up[i % (2 * n)] = 2;
    }else{
        int s_down = string_lower_find(x, x2);
        int st = s_down;
        FOR(i, st, st + SZ(done_list[2]))
            res_down[i % (2 * n)] = 2;
    }
    s_up = string_upper_find(x, x3, true);
    if (s_up >= 0){
        int st = s_up;
        FOR(i, st, st + SZ(done_list[3]))
            res_up[i % (2 * n)] = 3;
    }else{
        int s_down = string_lower_find(x, x3);
        int st = s_down;
        FOR(i, st, st + SZ(done_list[3]))
            res_down[i % (2 * n)] = 3;
    }
    REP(i, 2 * n){
        if (res_up[i] == -1){
            if (state[i] == x4)
                res_up[i] = 4;
            else if (state[i] == x5)
                res_up[i] = 5;
            else if (state[i] == x6)
                res_up[i] = 6;
            else if (state[i] == x7)
                res_up[i] = 7;
            else
                res_up[i] = 8;
        }
        if (res_down[i] == -1){
            if (state[i + 2 * n] == x4)
                res_down[i] = 4;
            else if (state[i + 2 * n] == x5)
                res_down[i] = 5;
            else if (state[i + 2 * n] == x6)
                res_down[i] = 6;
            else if (state[i + 2 * n] == x7)
                res_down[i] = 7;
            else
                res_down[i] = 8;
        }
    }
    // # print(res_up)
    // # print(res_down)
    VI length_list(max(MAX(res_up), MAX(res_down)) + 1, 0);
    for (int c: res_up)
        length_list[c] += 1;
    for (int c : res_down)
        length_list[c] += 1;
    FOR(i, 4, SZ(length_list))
        length_list[i] = 1;

    VVI new_goal_state;
    // VVI new_goal_state_sub;
    if (goal_state[4 * n - 2] == goal_state[4 * n - 1]){
        new_goal_state = {{0, 4, 1, 5}, {2, 6, 3, 7, 7}};
        // new_goal_state_sub = {{0, 4, 2, 6}, {1, 5, 3, 7, 7}};
    }else{
        new_goal_state = {{0, 4, 1, 5}, {2, 6, 3, 7, 8}};
        // new_goal_state_sub = {{0, 4, 2, 6}, {1, 5, 3, 7, 8}};
    }

    VVI initial_state{VI(), VI()};
    int r_0 = 0;
    if (res_up[0] == res_up.back() && res_up.back() < 4){
        REP(i, 2 * n)
            if (res_up[i] != res_up.back()){
                r_0 = i;
                break;
            }
    }
    int s = -1;
    FOR(i, r_0, SZ(res_up)){
        int c = res_up[i];
        if (c != s || c >= 4){
            s = c;
            initial_state[0].emplace_back(c);
        }
    }
    int r_1 = 0;
    if (res_down[0] == res_down.back() && res_down.back() < 4){
        REP(i, 2 * n)
            if (res_down[i] != res_down.back()){
                r_1 = i;
                break;
            }
    }
    s = -1;
    FOR(i, r_1, SZ(res_down)){
        int c = res_down[i];
        if (c != s || c >= 4){
            s = c;
            initial_state[1].emplace_back(c);
        }
    }
    // # print(initial_state)
    // # print(goal_state)
    // # print(length_list)
    // # print(r_0, r_1)
    return solve_greed(initial_state, new_goal_state, length_list, r_0, r_1);
}

VS solve_1xn(VI &initial_state, VI &goal_state, bool any_flip = false){
    int n = SZ(initial_state) / 4;
    map<string, VI> allowed_moves_mod;

    VI perm;
    FOR(i, 1, 2*n)perm.emplace_back(i);
    perm.emplace_back(0);
    FOR(i, 2*n, 4*n)perm.emplace_back(i);
    allowed_moves_mod["r0"] = perm;
    allowed_moves_mod["-r0"] = inverse(perm);

    perm.clear();
    REP(i, 2*n)perm.emplace_back(i);
    FOR(i, 2*n+1, 4*n)perm.emplace_back(i);
    perm.emplace_back(2*n);
    allowed_moves_mod["r1"] = perm;
    allowed_moves_mod["-r1"] = inverse(perm);

    perm.clear();
    RFOR(i, 2 * n, 3 * n)perm.emplace_back(i);
    FOR(i, n, 2 * n)perm.emplace_back(i);
    RFOR(i, 0, n)perm.emplace_back(i);
    FOR(i, 3*n, 4 * n)perm.emplace_back(i);
    allowed_moves_mod["f0"] = perm;

    perm.clear();
    FOR(i, 1, 2 * n){
        allowed_moves_mod["f"+to_string(i)] = do_action(do_action(do_action(do_action(allowed_moves_mod["-r0"], allowed_moves_mod["-r1"]), allowed_moves_mod["f"+to_string(i - 1)]), allowed_moves_mod["r0"]), allowed_moves_mod["r1"]);
    }

    VS sol;
    // # print(allowed_moves_mod)
    // VI i_list(2*n);
    vector<PII> i_list2(2*n);
    REP(i, 2*n)
        // i_list[i]=min(i % n, ((-i)%n+n) % n);
        i_list2[i]={min(i % n, ((-i)%n+n) % n), i};
    VI i_list = argsort(i_list2);
    if (any_flip)
        i_list.insert(i_list.begin(), -1);
    else
        i_list.emplace_back(-1);

    for (int i : i_list){
        VVI done_list{{goal_state[0]}, {goal_state[n]}, {goal_state[2 * n]}, {goal_state[3 * n]}};
        auto state = initial_state;
        VS sol_add;
        sol.clear();
        REP(k, n - 2){
            REP(j, 4){
                if (k >= n - 3 && j == 3)
                    break;
                tie(state, sol_add) = add_one(
                    state, goal_state, allowed_moves_mod, done_list,
                    j, goal_state[n * j + k + 1], i
                );
                sol.insert(sol.end(), sol_add.begin(), sol_add.end());
                done_list[j].emplace_back(goal_state[n * j + k + 1]);
                // # print(k, j, done_list)
            }
        }
        // # print(state)
        assert(heuristic_0(state, done_list) == 0);
        auto result = solve_last(state, goal_state, done_list);
        if (result){
            sol_add = result.value().second;
            sol.insert(sol.end(), sol_add.begin(), sol_add.end());
            OUT("Success at center", i);
            break;
        }else
            OUT("Failed at center", i);
    }
    return sol;
}

const string DATA_DIR = "./data/";
set<string> TARGET{
    //    "globe_1/8",
    //    "globe_1/16",
    //    "globe_2/6",
       "globe_3/4",
    //    "globe_6/4",
    //    "globe_6/8",
    //    "globe_6/10",
    //    "globe_3/33",
    //    "globe_8/25"
};

int main() {
    ChronoTimer timer;
    int case_num = 398;
    double sum_score = 0.0;
    double sum_log_score = 0.0;
    int64_t max_time = 0;
    FOR(i, 338, case_num){
        timer.start();
        dump(SEED)
        rand_engine.seed(SEED);
        string input_filename = to_string(i) + ".txt";
        string file_path = DATA_DIR + input_filename;
        ifstream ifs(file_path);
        assert(!ifs.fail());
        data_load(ifs);
        if(!TARGET.contains(puzzle_type)) continue;
        OUT("id", i);
        double score=0;
        string output_filename="globe/"+to_string(i)+".txt";

        ///////////////////////////////////////// 
        VS _sol_all;
        int _y=X, _n=Y;
        dump(X)
        dump(Y)
        dump(initial_state)
        dump(solution_state)
        REP(_j, (_y + 1) / 2){
            VI _initial_state, _goal_state;
            if (_j > 0){
                int _left = _j * (2 * _n);
                int _right = (_j + 1) * (2 * _n);
                // _initial_state = _initial_state_all[_left:_right] + _initial_state_all[-_right:-_left]
                _initial_state.insert(_initial_state.end(), initial_state.begin()+_left, initial_state.begin()+_right);
                for(int i=SZ(initial_state)-_right; i<SZ(initial_state)-_left; ++i)
                    _initial_state.emplace_back(initial_state[i]);
                // _goal_state = _goal_state_all[_left:_right] + _goal_state_all[-_right:-_left]
                _goal_state.insert(_goal_state.end(), solution_state.begin()+_left, solution_state.begin()+_right);
                for(int i=SZ(solution_state)-_right; i<SZ(solution_state)-_left; ++i)
                    _goal_state.emplace_back(solution_state[i]);
            }else{
                // _initial_state = _initial_state_all[:2 * _n] + _initial_state_all[-2 * _n:]
                _initial_state.insert(_initial_state.end(), initial_state.begin(), initial_state.begin()+2*_n);
                for(int i=SZ(initial_state)-2*_n; i<SZ(initial_state); ++i)
                    _initial_state.emplace_back(initial_state[i]);
                // _goal_state = _goal_state_all[:2 * _n] + _goal_state_all[-2 * _n:]
                _goal_state.insert(_goal_state.end(), solution_state.begin(), solution_state.begin()+2*_n);
                for(int i=SZ(solution_state)-2*_n; i<SZ(solution_state); ++i)
                    _goal_state.emplace_back(solution_state[i]);
            }
            OUT("sub problem:", _j);
            OUT("initial_state:", _initial_state);
            OUT("goal_state:", _goal_state);

            auto _sol = solve_1xn(_initial_state, _goal_state);
            // auto _sol = solve_1xn(_initial_state, _goal_state, true);
            dump(_sol)
            for (auto &_m :_sol){
                if (_m[0] == 'f')
                    _sol_all.emplace_back(_m);
                else if (_m == "r0")
                    _sol_all.emplace_back("r"+to_string(_j));
                else if (_m == "-r0")
                    _sol_all.emplace_back("-r"+to_string(_j));
                else if (_m == "r1")
                    _sol_all.emplace_back("r"+to_string(_y - _j));
                else if (_m == "-r1")
                    _sol_all.emplace_back("-r"+to_string(_y - _j));
            }
            VI candidate(2*_n+1);
            candidate[0]=_n;
            REP(i,2*_n)candidate[i+1]=i;
            for (auto _q : candidate){
                if (std::count(ALL(_sol_all), "f"+to_string(_q)) % 2 == 1){
                    REP(_, _n)
                        _sol_all.emplace_back("r"+to_string(_j));
                    _sol_all.emplace_back("f"+to_string(_q));
                    REP(_, _n)
                        _sol_all.emplace_back("r"+to_string(_j));
                    _sol_all.emplace_back("f"+to_string(_q));
                    REP(_, _n)
                        _sol_all.emplace_back("r"+to_string(_j));
                    _sol_all.emplace_back("f"+to_string(_q));
                }
            }
        }
        ofstream ofs(output_filename);
        REP(i, SZ(_sol_all)-1)
            ofs<<_sol_all[i]<<".";
        assert(!_sol_all.empty());
        ofs<<_sol_all.back();
        ofs.close();
        // 

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
