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
    T hash(const VI &array, const SI& target) {
        T value = 0;
        REP(i, size){
            // (i, e) → i*nunique + e
            auto &e=array[i];
            if(target.contains(e))
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

VI do_action(const VI& state, const unordered_map<int,int> &action){
    auto s=state;
    for(auto [i, v]: action)
        s[i]=state[v];
    return s;
}

VI do_action(const VI& state, const VI &action){
    auto s=state;
    REP(i, SZ(state))
        s[i]=state[action[i]];
    return s;
}

VI product(const VI& perm0, const VI &perm1){
    return do_action(perm1, perm0);
}
VI product(const VI& perm0, const unordered_map<int,int> &perm1){
    VI perm1_vec(state_length);
    REP(i, state_length){
        auto it1=perm1.find(i);
        perm1_vec[i]=it1==perm1.end()?i:it1->second;
    }
    return do_action(perm1_vec, perm0);
}
VI product(const unordered_map<int,int>& perm0, const VI &perm1){
    return do_action(perm1, perm0);
}

VI product(const unordered_map<int,int> &perm0, const unordered_map<int,int> &perm1){
    // VI perm0_vec(state_length);
    VI perm1_vec(state_length);
    REP(i, state_length){
        // auto it0=perm0.find(i);
        // perm0_vec[i]=it0==perm0.end()?i:it0->second;
        auto it1=perm1.find(i);
        perm1_vec[i]=it1==perm1.end()?i:it1->second;
    }
    return do_action(perm1_vec, perm0);
}

VI product(const VI& perm, int a){
    return product(perm, get_action(a));
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

VI simulation_inverse(const VI& state, const VI &actions){
    auto s=state;
    RITR(it, actions)
        s = do_action(s, inverse(*it));
    return s;
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

VI construct_actions(uint64_t init_hash, uint64_t last_hash, const unordered_map<uint64_t, tuple<uint64_t,int>> &pushed){
    VI actions;
    auto h=last_hash;
    while(h!=init_hash){
        const auto &[next,a]=pushed.at(h);
        // const auto &[next,a]=pushed.at(h);
        if(a>=0)actions.emplace_back(a);
        h=next;
    }
    REVERSE(actions);
    return actions;
}



// VI solve_center(){
//     int size=X*X;
//     auto center_penalty = [&](int panel, const VI& state, int intercept){
//         int penalty=0;
//         FOR(i, intercept, X-intercept)FOR(j, intercept, X-intercept){
//             int idx=size*panel+i*X+j;
//             penalty+=state[idx]!=solution_state[idx];
//         }
//         if(penalty%2==1)
//             penalty++;
//         penalty/=2;
//         return penalty;
//     };

//     ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);

//     VI target_panel;
//     SI target_label;
//     auto center_penalty_cumsum = [&](const VI& state, int intercept){
//         int penalty=0;
//         for(int t: target_panel)
//             penalty+=center_penalty(t, state, intercept);
//         return penalty;
//     };

//     function<bool(const VI &, VI &, int, int, unordered_set<uint64_t> &, int)> search;
//     search=[&](const VI &state, VI &path, int same_action_num, int max_action_size, unordered_set<uint64_t> &visit, int intercept){
//         int h=center_penalty_cumsum(state, intercept);
//         if (h==0)
//             return true;
//         // if(SZ(path) >= max_action_size) return false;
//         if(h+SZ(path) >= max_action_size) return false;
//         REP(a, allowed_action_num*2){
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
//             auto next=do_action(state, a);
//             auto next_hash=zhash.hash(next, target_label);
//             if (visit.contains(next_hash))
//                 continue;
//             visit.emplace(next_hash);
//             path.emplace_back(a);
//             if (search(next, path, next_same_action_num, max_action_size, visit, intercept))
//                 return true;
//             path.pop_back();
//             visit.erase(next_hash);
//         }
//         return false;
//     };


//     auto state=initial_state;
//     VI all_path;
//     VI panels{0,5, 1,2, 4,3};
//     shuffle(ALL(panels), rand_engine);
//     RFOR(intercept, 1, X/2){
//         for(auto panel: panels){
//             target_panel.emplace_back(panel);

//             dump(intercept)
//             FOR(i, intercept, X-intercept)FOR(j, intercept, X-intercept){
//                 int idx=size*panel+i*X+j;
//                 target_label.emplace(solution_state[idx]);
//             }
//             FOR(max_action_size, 1, INF){
//                 dump(max_action_size)
//                 unordered_set<uint64_t> visit;
//                 visit.emplace(zhash.hash(state, target_label));
//                 VI path;
//                 if(search(state, path, 0, max_action_size, visit, intercept)){
//                     OUT("Find!", path, target_panel);
//                     dump(action_decode(path))
//                     state=simulation(state, path);
//                     all_path.insert(all_path.end(), path.begin(), path.end());
//                     break;
//                 }
//             }
//         }
//         // // hash -> {prev_hash, action_id}
//         // unordered_map<uint64_t, tuple<uint64_t,int>> pushed;
//         // // mistake, length, hash
//         // MINPQ<tuple<int, int, uint64_t>> pq;
//         // auto init_hash = zhash.hash(state, target_label);
//         // pushed[init_hash]={0, -1};
//         // int p=center_penalty_cumsum(state);
//         // pq.emplace(p, p, init_hash);
//         // int searched=0;
//         // while(!pq.empty()){
//         //     auto [_, mistake, hash] = pq.top(); pq.pop();
//         //     searched++;
//         //     if(searched%100000==0)
//         //         dump(searched)
//         //     auto actions=construct_actions(init_hash, hash, pushed);
//         //     auto current_state=simulation(state, actions);
//         //     if(mistake==0){
//         //         // OUT("find", panel1, panel2, searched);
//         //         OUT("find", panel, searched);
//         //         state=current_state;
//         //         break;
//         //     }
//         //     REP(a, allowed_action_num*2){
//         //         auto next = do_action(current_state, a);
//         //         auto next_hash = zhash.hash(next);
//         //         if(pushed.contains(next_hash))
//         //             continue;
//         //         pushed[next_hash]={hash, a};
//         //         p=center_penalty_cumsum(next);
//         //         pq.emplace(p+SZ(actions)+1, p, next_hash);
//         //         if(pq.size()%5000000==0)
//         //             dump(pq.size())
//         //         if(pushed.size()%5000000==0)
//         //             dump(pushed.size())
//         //     }
//         // }
//     }
//     return all_path;
// }

VI solve_center2(const VI& start){
    int size=X*X;
    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);

    SI target_label;

    auto center_penalty_cumsum = [&](const VI& state, int center_num, int intercept){
        int ok_c=0;
        REP(t, 6){
            int row=0, col=0;
            FOR(i, intercept, X-intercept){
                bool ok=true;
                FOR(j, intercept, X-intercept)if(ok){
                    int idx=size*t+i*X+j;
                    if(state[idx]!=solution_state[idx])ok=false;
                }
                row+=ok;
                ok=true;
                FOR(j, intercept, X-intercept)if(ok){
                    int idx=size*t+j*X+i;
                    if(state[idx]!=solution_state[idx])ok=false;
                }
                col+=ok;
            }
            ok_c+=max(row, col);
        }
        return max(0, (X-2*intercept)*center_num-ok_c);
    };

    VI allowed(allowed_action_num*2);
    ARANGE(allowed);
    shuffle(ALL(allowed), rand_engine);

    function<bool(const VI &, VI &, int, int, unordered_set<uint64_t> &, int, int)> search;
    search=[&](const VI &state, VI &path, int same_action_num, int max_action_size, unordered_set<uint64_t> &visit, int center_num, int intercept){
        int h=center_penalty_cumsum(state, center_num, intercept);
        if (h==0)
            return true;
        // if(SZ(path) >= max_action_size) return false;
        if(h+SZ(path) >= max_action_size) return false;
        // REP(a, allowed_action_num*2){
        for(int a : allowed){
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
            auto next_hash=zhash.hash(next, target_label);
            if (visit.contains(next_hash))
                continue;
            visit.emplace(next_hash);
            path.emplace_back(a);
            if (search(next, path, next_same_action_num, max_action_size, visit, center_num, intercept))
                return true;
            path.pop_back();
            visit.erase(next_hash);
        }
        return false;
    };


    auto state=start;
    VI all_path;
    RREP(intercept, X/2){
        dump(intercept)
        REP(t, 6){
            FOR(i, intercept, X-intercept){
                FOR(j, intercept, X-intercept){
                    target_label.emplace(solution_state[size*t+i*X+j]);
                }
            }
        }
        FOR(center_num, 1, 6+1){
            FOR(max_action_size, 1, INF){
                dump(max_action_size)
                unordered_set<uint64_t> visit;
                visit.emplace(zhash.hash(state, target_label));
                VI path;
                if(search(state, path, 0, max_action_size, visit, center_num, intercept)){
                    OUT("Find!", path, center_num);
                    dump(action_decode(path))
                    state=simulation(state, path);
                    all_path.insert(all_path.end(), path.begin(), path.end());
                    break;
                }
            }
        }
        // break;
        break;
    }
    return all_path;
}

VI solve_center_corner(const VI& start){
    int size=X*X;
    
    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);

    SI target_label;
    VVI rotated_panel;
    rotated_panel.emplace_back(solution_state);
    REP(r,3){
        auto state=solution_state;
        auto &prev=rotated_panel.back();
        REP(t, 6)REP(i,X)REP(j,X){
            state[size*t+j*X+i]=prev[size*t+(X-1-i)*X+j];
        }
        rotated_panel.emplace_back(state);
    }

    // auto center_penalty_cumsum = [&](const VI& state, int center_num, int intercept){
    //     int penalty=0;
    //     FOR(inter, intercept, X/2){
    //         int ok_c=0;
    //         REP(t, 6){
    //             int row=0, col=0;
    //             FOR(i, inter, X-inter){
    //                 bool ok=true;
    //                 FOR(j, inter, X-inter)if(ok){
    //                     int idx=size*t+i*X+j;
    //                     if(state[idx]!=solution_state[idx])ok=false;
    //                 }
    //                 row+=ok;
    //                 ok=true;
    //                 FOR(j, inter, X-inter)if(ok){
    //                     int idx=size*t+j*X+i;
    //                     if(state[idx]!=solution_state[idx])ok=false;
    //                 }
    //                 col+=ok;
    //             }
    //             ok_c+=max(row, col);
    //         }
    //         penalty+=max(0, (X-2*inter)*center_num-ok_c);
    //     }
    //     return penalty;
    // };
    // auto center_penalty_cumsum = [&](const VI& state, int center_num, int intercept){
    //     int corner=0;
    //     int ok_c=0;
    //     REP(t, 6){
    //         int row=0, col=0;
    //         FOR(i, intercept, X-intercept){
    //             bool ok=true;
    //             FOR(j, intercept, X-intercept)if(ok){
    //                 int idx=size*t+i*X+j;
    //                 if(state[idx]!=solution_state[idx])ok=false;
    //             }
    //             row+=ok;
    //             ok=true;
    //             FOR(j, intercept, X-intercept)if(ok){
    //                 int idx=size*t+j*X+i;
    //                 if(state[idx]!=solution_state[idx])ok=false;
    //             }
    //             col+=ok;
    //         }
    //         ok_c+=max(row, col);
    //         // 
    //         for(int i: VI{1, X-1})for(int j: VI{1, X-1}){
    //             int idx=size*t+i*X+j;
    //             if(state[idx]==solution_state[idx])
    //                 corner++;
    //         }
    //     }
    //     return max(0, (X-2*intercept)*center_num-ok_c)+max(0, 4*center_num-corner);
    // };
    auto center_penalty_cumsum = [&](const VI& state, int edge_num, int intercept){
        // int corner=0;
        int edge=0;
        // int ok_c=0;
        REP(t, 6){
            // int row=0, col=0;
            // center
            // for(int i:VI{X/2-1, X/2}){
            //     bool ok=true;
            //     for(int j:VI{X/2-1, X/2})if(ok){
            //         int idx=size*t+i*X+j;
            //         if(state[idx]!=solution_state[idx])ok=false;
            //     }
            //     row+=ok;
            //     ok=true;
            //     for(int j:VI{X/2-1, X/2})if(ok){
            //         int idx=size*t+j*X+i;
            //         if(state[idx]!=solution_state[idx])ok=false;
            //     }
            //     col+=ok;
            // }
            // ok_c+=max(row, col);
            // 
            // for(int i: VI{1, X-1})for(int j: VI{1, X-1}){
            //     int idx=size*t+i*X+j;
            //     if(state[idx]==solution_state[idx])
            //         corner++;
            // }
            //
            FOR(inter, intercept, X/2){
                // int row=0, col=0;
                // for(int i:VI{X/2-1, X/2}){
                //     bool ok=true;
                //     for(int j: VI{inter, X-1-inter})if(ok){
                //         int idx=size*t+i*X+j;
                //         if(state[idx]!=solution_state[idx])ok=false;
                //     }
                //     row+=ok;

                //     ok=true;
                //     for(int j: VI{inter, X-1-inter})if(ok){
                //         int idx=size*t+j*X+i;
                //         if(state[idx]!=solution_state[idx])ok=false;
                //     }
                //     col+=ok;
                // }
                // edge+=(row + col);
                int emax=0;
                for(auto & target: rotated_panel){
                    int e=0;
                    for(int j: VI{inter, X-1-inter}){
                        bool ok=true;
                        for(int i:VI{X/2-1, X/2})if(ok){
                            int idx=size*t+i*X+j;
                            if(state[idx]!=target[idx])ok=false;
                        }
                        e+=ok;
                    }
                    for(int i: VI{inter, X-1-inter}){
                        bool ok=true;
                        for(int j:VI{X/2-1, X/2})if(ok){
                            int idx=size*t+i*X+j;
                            if(state[idx]!=target[idx])ok=false;
                        }
                        e+=ok;
                    }
                    // for(int j: VI{inter, X-1-inter}){
                    //     for(int i:VI{X/2-1, X/2}){
                    //         int idx=size*t+i*X+j;
                    //         if(state[idx]==target[idx])e++;
                    //     }
                    // }
                    // for(int i: VI{inter, X-1-inter}){
                    //     for(int j:VI{X/2-1, X/2}){
                    //         int idx=size*t+i*X+j;
                    //         if(state[idx]==target[idx])e++;
                    //     }
                    // }
                    emax=max(emax,e);
                }
                edge+=emax;
            }
        }
        return max(0, edge_num-edge);
    };

    VI allowed(allowed_action_num*2);
    ARANGE(allowed);
    shuffle(ALL(allowed), rand_engine);

    function<bool(const VI &, VI &, int, int, unordered_set<uint64_t> &, int, int)> search;
    search=[&](const VI &state, VI &path, int same_action_num, int max_action_size, unordered_set<uint64_t> &visit, int edge_num, int intercept){
        int h=center_penalty_cumsum(state, edge_num, intercept);
        if (h==0)
            return true;
        // if(SZ(path) >= max_action_size) return false;
        if(h+SZ(path) >= max_action_size) return false;
        // REP(a, allowed_action_num*2){
        for(int a : allowed){
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
            auto next_hash=zhash.hash(next, target_label);
            if (visit.contains(next_hash))
                continue;
            visit.emplace(next_hash);
            path.emplace_back(a);
            if (search(next, path, next_same_action_num, max_action_size, visit, edge_num, intercept))
                return true;
            path.pop_back();
            visit.erase(next_hash);
        }
        return false;
    };


    auto state=start;
    VI all_path;
    int total_edge_num=0;
    RREP(intercept, X/2){
        dump(intercept)
        // REP(t, 6){
        //     FOR(i, intercept, X-intercept){
        //         FOR(j, intercept, X-intercept){
        //             target_label.emplace(solution_state[size*t+i*X+j]);
        //         }
        //     }
        //     for(int i: VI{1, X-1})for(int j: VI{1, X-1})
        //         target_label.emplace(solution_state[size*t+i*X+j]);
        // }
        REP(t,6)
        for(int i:VI{X/2-1, X/2}){
            for(int j: VI{intercept, X-1-intercept}){
                target_label.emplace(solution_state[size*t+i*X+j]);
                target_label.emplace(solution_state[size*t+j*X+i]);
            }
        }
        REP(edge_loop, 4*6){
        // REP(point_loop, 8*6){
            total_edge_num++;
            FOR(max_action_size, 1, INF){
                dump(max_action_size)
                unordered_set<uint64_t> visit;
                visit.emplace(zhash.hash(state, target_label));
                VI path;
                if(search(state, path, 0, max_action_size, visit, total_edge_num, intercept)){
                    OUT("Find!", path, total_edge_num);
                    dump(action_decode(path))
                    state=simulation(state, path);
                    all_path.insert(all_path.end(), path.begin(), path.end());
                    dump(action_decode(all_path))
                    break;
                }
            }
        }
        // break;
    }
    return all_path;
}


// VI solve_edge(const VI& start){
//     int size=X*X;
//     auto center_penalty = [&](int panel, const VI& state){
//         int penalty=0;
//         FOR(i, 1, X-1)FOR(j, 1, X-1){
//             int idx=size*panel+i*X+j;
//             penalty+=state[idx]!=solution_state[idx];
//         }
//         if(penalty%2==1)
//             penalty++;
//         penalty/=2;
//         return penalty;
//     };
//     map<VI, int> edges;
//     REP(panel, 6)for(int j: VI{0, X-1}){
//         VI e1,e2;
//         FOR(i, 1, X-1){
//             e1.emplace_back(solution_state[size*panel+i*X+j]);
//             e2.emplace_back(solution_state[size*panel+j*X+i]);
//         }
//         edges[e1]++;
//         edges[e2]++;
//         // edges.emplace_back(e1);
//         // edges.emplace_back(e2);
//     }

//     auto edge_ok = [&](int panel, const VI& state){
//         int ok=0;
//         for(int j: VI{0, X-1}){
//             VI e1,e2;
//             FOR(i, 1, X-1){
//                 e1.emplace_back(state[size*panel+i*X+j]);
//                 e2.emplace_back(state[size*panel+j*X+i]);
//             }
//             if(edges.contains(e1))
//                 ok++;
//             if(edges.contains(e2))
//                 ok++;
//         }
//         return ok;
//     };

//     ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);

//     auto center_penalty_cumsum = [&](const VI& state, int edge_num){
//         int penalty=0;
//         int ok_e=0;
//         REP(t, 6){
//             penalty+=center_penalty(t, state);
//             ok_e+=edge_ok(t, state);
//         }
//         return penalty+max(0, edge_num-ok_e);
//     };

//     function<bool(const VI &, VI &, int, int, unordered_set<uint64_t> &, int)> search;
//     search=[&](const VI &state, VI &path, int same_action_num, int max_action_size, unordered_set<uint64_t> &visit, int edge_num){
//         int h=center_penalty_cumsum(state, edge_num);
//         if (h==0)
//             return true;
//         // if(SZ(path) >= max_action_size) return false;
//         if(h+SZ(path) >= max_action_size) return false;
//         REP(a, allowed_action_num*2){
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
//             auto next=do_action(state, a);
//             auto next_hash=zhash.hash(next);
//             if (visit.contains(next_hash))
//                 continue;
//             visit.emplace(next_hash);
//             path.emplace_back(a);
//             if (search(next, path, next_same_action_num, max_action_size, visit, edge_num))
//                 return true;
//             path.pop_back();
//             visit.erase(next_hash);
//         }
//         return false;
//     };

//     auto state=start;
//     VI all_path;
//     FOR(i, 1, 6*4+1){
//         FOR(max_action_size, 1, INF){
//             dump(max_action_size)
//             unordered_set<uint64_t> visit;
//             visit.emplace(zhash.hash(state));
//             VI path;
//             if(search(state, path, 0, max_action_size, visit, i)){
//                 OUT("Find!", path, i);
//                 state=simulation(state, path);
//                 all_path.insert(all_path.end(), path.begin(), path.end());
//                 break;
//             }
//         }
//     }
//     return all_path;
// }


VI solve_edge_correct(const VI& start){
    int size=X*X;

    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);
    SI target_label;
    REP(t, 6){
        FOR(i, 1, X-1)FOR(j, 1, X-1)
            target_label.emplace(solution_state[size*t+i*X+j]);
        for(int j: VI{0, X-1}){
            FOR(i, 1, X-1){
                target_label.emplace(solution_state[size*t+i*X+j]);
                target_label.emplace(solution_state[size*t+j*X+i]);
            }
        }
    }

    auto center_penalty_cumsum = [&](const VI& state, int edge_num){
        int ok_c=0;
        int ok_e=0;
        REP(t, 6){
            int row=0, col=0;
            FOR(i, 1, X-1){
                bool ok=true;
                FOR(j, 1, X-1)if(ok){
                    int idx=size*t+i*X+j;
                    if(state[idx]!=solution_state[idx])ok=false;
                }
                row+=ok;
                ok=true;
                FOR(j, 1, X-1)if(ok){
                    int idx=size*t+j*X+i;
                    if(state[idx]!=solution_state[idx])ok=false;
                }
                col+=ok;
            }
            ok_c+=max(row, col);

            for(int j: VI{0, X-1}){
                bool ok=true;
                FOR(i, 1, X-1)if(state[size*t+i*X+j]!=solution_state[size*t+i*X+j]){
                    ok=false;break;
                }
                ok_e+=ok;
                ok=true;
                FOR(i, 1, X-1)if(state[size*t+j*X+i]!=solution_state[size*t+j*X+i]){
                    ok=false;break;
                }
                ok_e+=ok;
            }
        }
        return max(0, (X-2)*6-ok_c)+max(0, edge_num-ok_e);
    };


    VI allowed(allowed_action_num*2);
    ARANGE(allowed);
    shuffle(ALL(allowed), rand_engine);

    function<bool(const VI &, VI &, int, int, unordered_set<uint64_t> &, int)> search;
    search=[&](const VI &state, VI &path, int same_action_num, int max_action_size, unordered_set<uint64_t> &visit, int edge_num){
        int h=center_penalty_cumsum(state, edge_num);
        if (h==0)
            return true;
        // if(SZ(path) >= max_action_size) return false;
        if(h+SZ(path) >= max_action_size) return false;
        // REP(a, allowed_action_num*2){
        for(int a : allowed){
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
            auto next_hash=zhash.hash(next, target_label);
            if (visit.contains(next_hash))
                continue;
            visit.emplace(next_hash);
            path.emplace_back(a);
            if (search(next, path, next_same_action_num, max_action_size, visit, edge_num))
                return true;
            path.pop_back();
            visit.erase(next_hash);
        }
        return false;
    };


    auto state=start;
    VI all_path;
    FOR(edge_num, 1, 24+1){
        FOR(max_action_size, 1, INF){
            dump(max_action_size)
            unordered_set<uint64_t> visit;
            visit.emplace(zhash.hash(state, target_label));
            VI path;
            if(search(state, path, 0, max_action_size, visit, edge_num)){
                // OUT("Find!", path, target_panel);
                OUT("Find!", path, edge_num);
                dump(action_decode(path))
                state=simulation(state, path);
                all_path.insert(all_path.end(), path.begin(), path.end());
                break;
            }
        }
    }
    return all_path;
}

int cube_solver(){
    // auto center_path=solve_center();
    // auto state=simulation(initial_state, action_encode("f3.r2.-d3.-f5.r5.d3.r0.-r5.-d3.f0.-r0.d0.-f2.-d3.f2.r0.-f5.d0.f5.r2.d0.d0.-r2.-d3.f5.d3.f5.r5.d3.f2.-r5.d3.d3.-f2.d3.f2.r5.-f2.-r5.r5.d5.f2.-d5.-f2.d5.f3.-d5.-f3.-d5"));
    auto state=simulation(initial_state, action_encode("-d2.-f3.d2.-f5.-d3.f2.-d3.-f3.-d3.-d2.-r2.d0.d0.r2.d2.-r0.-d2.f0.f5.f5.d2.-d3.-r0.-r5.d3.-r0.-f3.-d3.r0.d3.f3.r0.-f5.r2.-f5.-r2.f5.-f3.r0.-f3.-r2.-f0.r2.f3.-r0.f3.-r2.f0.r2"));
    // auto center_path=solve_center2(initial_state);
    auto center_path=solve_center_corner(state);
    dump(center_path);
    dump(action_decode(center_path));

    // auto state=simulation(initial_state, center_path);
    state=simulation(state, center_path);
    // auto edge_path=solve_edge(state);
    auto edge_path=solve_edge_correct(state);
    dump(edge_path);
    dump(action_decode(edge_path));
    return INF;
}




// センターを揃える
// https://cube.uubio.com/4x4x4/
VVI get_magic4_1(){
    // r2.d5.-r2
    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);

    for(const auto& d1: VS{"r","d","f"})FOR(i1, 1, X-1)REP(sign1, 2){
        const auto a1=(sign1?"":"-")+d1+to_string(i1);
        const auto a1inv=(sign1?"-":"")+d1+to_string(i1);
        for(const auto& d2: VS{"r","d","f"})if(d1!=d2)for(int i2: VI{0, X-1})REP(sign2, 2){
            const auto a2=(sign2?"":"-")+d2+to_string(i2);
            auto action=action_encode(a1+"."+a2+"."+a1inv);

            auto state=simulation(index, action);
            auto hash=zhash.hash(state);
            if(!ops.contains(hash) || action.size()<ops[hash].size())
                ops[hash]=action;

            // 180度
            action=action_encode(a1+"."+a1+"."+a2+"."+a1inv+"."+a1inv);
            state=simulation(index, action);
            hash=zhash.hash(state);
            if(!ops.contains(hash) || action.size()<ops[hash].size())
                ops[hash]=action;
        }
    }
    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}


VVI get_centers(){
    VVI centers;
    VI index(state_length);
    ARANGE(index);
    centers.emplace_back(index);
    REP(r,3){
        auto state=index;
        auto &prev=centers.back();
        REP(t, 6)REP(i,X)REP(j,X){
            state[X*X*t+j*X+i]=prev[X*X*t+(X-1-i)*X+j];
        }
        centers.emplace_back(state);
    }
    return centers;
}
bool check_center_change(const VVI &centers, const VI &state){
    bool allok=true;
    REP(t, 6)if(allok){
        bool okpanel=false;
        for(auto& center: centers)if(!okpanel){
            bool ok=true;
            FOR(i, 1, X-1)if(ok)FOR(j, 1, X-1)if(ok){
                int idx=X*X*t+i*X+j;
                if(state[idx]!=center[idx])ok=false;
            }
            okpanel|=ok;
        }
        allok&=okpanel;
    }
    return allok;
};


// 中央エッジを揃えるのに使う
// TODO: エッジだけが変更されていることの確認
VVI get_magic4_2(){
    // -r4.-d5.r5.d5.r4
    // -r{中央}.-d{N-1/0}.r{N-1/0}.d{N-1/0}.r{中央}
    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);

    VVI centers=get_centers();
    for(const auto &d1: VS{"r","d","f"})FOR(i1, 1, X-1)REP(sign1, 2)REP(cnt1, 2){
        auto a1=(sign1?"":"-")+d1+to_string(i1);
        if(cnt1)a1+="."+a1;

        for(const auto &d2: VS{"r","d","f"})if(d1!=d2)for(int i2: VI{0, X-1})REP(sign2, 2)REP(cnt2, 2){
            auto a2=(sign2?"":"-")+d2+to_string(i2);
            if(cnt2)a2+="."+a2;

            auto fronts=action_encode(a1+"."+a2);
            auto backs=inverse_action(fronts);

            for(int i3: VI{0, X-1})REP(sign3, 2)REP(cnt3, 2){
                auto a3=(sign3?"":"-")+d1+to_string(i3);
                if(cnt3)a3+="."+a3;

                auto action=fronts;
                for(int a: action_encode(a3))
                    action.emplace_back(a);
                action.insert(action.end(), backs.begin(), backs.end());

                auto state=simulation(index, action);
                auto hash=zhash.hash(state);
                bool allok=check_center_change(centers, state);
                if(allok && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;
            }
        }
    }
    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}

VVI get_magic4_3(){
    // r2.f5.f5.r3.f5.f5.-r3.f5.f5.
    // -r2.f5.f5.r3.f5.f5.-r3.f5.f5
    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);

    VVI centers=get_centers();
    for(const auto &d1: VS{"r","d","f"})FOR(i1, 1, X-1)REP(sign1, 2){
        auto a1=(sign1?"":"-")+d1+to_string(i1);
        auto a1inv=(sign1?"-":"")+d1+to_string(i1);
        int i2=X-1-i1;
        REP(sign2, 2){
            auto a2=(sign2?"":"-")+d1+to_string(i2);
            auto a2inv=(sign2?"-":"")+d1+to_string(i2);
            for(const auto & d3: VS{"r","d","f"})if(d1!=d3)for(int i3: VI{0, X-1}){
                const auto a3a3=d3+to_string(i3)+"."+d3+to_string(i3);
                auto tmp=a3a3+"."+a2+"."+a3a3+"."+a2inv+"."+a3a3;

                auto action=action_encode(a1+"."+tmp+"."+a1inv +"."+ tmp);
                auto state=simulation(index, action);
                auto hash=zhash.hash(state);
                bool allok=check_center_change(centers, state);
                if(allok && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;

                action=action_encode(a1+"."+tmp+"."+a1inv +"."+ tmp+"."+a1+"."+tmp+"."+a1inv +"."+ tmp);
                state=simulation(index, action);
                hash=zhash.hash(state);
                allok=check_center_change(centers, state);
                if(allok && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;
            }
        }
    }
    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}

VVI get_magic4_4(){
    // r2.r2.-f1.-f2.-f3.-f4.
    // -r2.d5.d5.-r2.d5.d5.-r2.d5.d5.-r2.d5.d5.-r2.
    // f4.f3.f2.f1.-r2.-r2
    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);

    VVI centers=get_centers();
    for(const auto & d1: VS{"r","d","f"})FOR(i1, 1, X-1)REP(sign1, 2){
        const auto a1=(sign1?"":"-")+d1+to_string(i1);
        const auto a1inv=(sign1?"-":"")+d1+to_string(i1);
        for(const auto & d2: VS{"r","d","f"})if(d1!=d2)REP(sign2, 2){
            auto a2=(sign2?"":"-")+d2+to_string(1);
            FOR(i2,2,X-1)
                a2=a2+string(".")+(sign2?"":"-")+d2+to_string(i2);
            auto fronts=action_encode(a1+"."+a1+"."+a2);
            auto backs=inverse_action(fronts);
            for(const auto & d3: VS{"r","d","f"})if(d1!=d3&&d2!=d3)for(int i3: VI{0, X-1}){
                const auto a3a3=d3+to_string(i3)+"."+d3+to_string(i3);
                auto tmp=a1inv+"."+a3a3;
                auto action=fronts;
                for(int a: action_encode(tmp+"."+tmp+"."+tmp+"."+tmp+"."+a1inv))
                    action.emplace_back(a);
                action.insert(action.end(), backs.begin(), backs.end());

                auto state=simulation(index, action);
                auto hash=zhash.hash(state);
                bool allok=check_center_change(centers, state);
                if(allok && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;

                action=fronts;
                for(int a: action_encode(tmp+"."+tmp+"."+tmp+"."+tmp+"."+a1inv+"."+tmp+"."+tmp+"."+tmp+"."+tmp+"."+a1inv))
                    action.emplace_back(a);
                action.insert(action.end(), backs.begin(), backs.end());

                state=simulation(index, action);
                hash=zhash.hash(state);
                allok=check_center_change(centers, state);
                if(allok && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;
            }
        }
    }
    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}

// センターを揃える
VVI get_magic5_1(){
    // r1.-d5.r4.d5.-r1.-d5.-r4.d5

    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);
    for(const auto & d1: VS{"r","d","f"})FOR(i1, 1, X-1)REP(sign1, 2){
        const auto a1=(sign1?"":"-")+d1+to_string(i1);
        const auto a1inv=(sign1?"-":"")+d1+to_string(i1);
        for(const auto & d2: VS{"r","d","f"})FOR(i2, 1, X-1)REP(sign2, 2){
            const auto a2=(sign2?"":"-")+d2+to_string(i2);
            const auto a2inv=(sign2?"-":"")+d2+to_string(i2);
            for(const auto & d3: VS{"r","d","f"})for(int i3: VI{0, X-1})REP(sign3, 2){
                // if(d1==d3) continue;
                const auto a3=(sign3?"":"-")+d3+to_string(i3);
                const auto a3inv=(sign3?"-":"")+d3+to_string(i3);

                auto action=action_encode(a1+"."+a3inv+"."+a2+"."+a3+"."+a1inv+"."+a3inv+"."+a2inv+"."+a3);
                auto state=simulation(index, action);
                int mistake=get_mistakes(state, index);
                auto hash=zhash.hash(state);
                if(mistake==3 && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;

                action=action_encode(a1+"."+a1+"."+a3inv+"."+a2+"."+a3+"."+a1inv+"."+a1inv+"."+a3inv+"."+a2inv+"."+a3);
                state=simulation(index, action);
                mistake=get_mistakes(state, index);
                hash=zhash.hash(state);
                if(mistake==3 && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;
                
                action=action_encode(a1+"."+a3inv+"."+a2+"."+a2+"."+a3+"."+a1inv+"."+a3inv+"."+a2inv+"."+a2inv+"."+a3);
                state=simulation(index, action);
                mistake=get_mistakes(state, index);
                hash=zhash.hash(state);
                if(mistake==3 && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;
                
                action=action_encode(a1+"."+a1+"."+a3inv+"."+a2+"."+a2+"."+a3+"."+a1inv+"."+a1inv+"."+a3inv+"."+a2inv+"."+a2inv+"."+a3);
                state=simulation(index, action);
                mistake=get_mistakes(state, index);
                hash=zhash.hash(state);
                if(mistake==3 && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;
            }
        }
    }
    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}
VVI get_magic5_2(){
    // -r2.d5.r0.-d5.r2
    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);
    VVI centers=get_centers();
    for(const auto & d1: VS{"r","d","f"})FOR(i1, 1, X-1)REP(sign1, 2)REP(cnt1, 2){
        auto a1=(sign1?"":"-")+d1+to_string(i1);
        if(cnt1)a1+="."+a1;
        auto a1inv=(sign1?"-":"")+d1+to_string(i1);
        if(cnt1)a1inv+="."+a1inv;
        for(const auto & d2: VS{"r","d","f"})for(int i2: VI{0, X-1})REP(sign2, 2)REP(cnt2, 2){
            auto a2=(sign2?"":"-")+d2+to_string(i2);
            if(cnt2)a2+="."+a2;
            auto a2inv=(sign2?"-":"")+d2+to_string(i2);
            if(cnt2)a2inv+="."+a2inv;
            for(int i3: VI{0, X-1})REP(sign3, 2)REP(cnt3, 2){
                auto a3=(sign3?"":"-")+d1+to_string(i3);
                if(cnt3)a3+="."+a3;
                auto a3inv=(sign3?"-":"")+d1+to_string(i3);
                if(cnt3)a3inv+="."+a3inv;

                auto action=action_encode(a1+"."+a3+"."+a2+"."+a3inv+"."+a1inv);
                auto state=simulation(index, action);
                auto hash=zhash.hash(state);
                bool allok=check_center_change(centers, state);
                if(allok && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;
            }
        }
    }
    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}

VVI get_help_magic(){
    // "f5.f5.-d0.-r0.-f5"
    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);
    for(const auto &d1: VS{"r","d","f"})for(int i1: VI{0, X-1})for(int pm1: VI{0, 1}){
        const auto d1_str=(pm1?"":"-")+d1+to_string(i1);
        const auto d1_strinv=(pm1?"-":"")+d1+to_string(i1);
        for(const auto &d2: VS{"r","d","f"})for(int i2: VI{0, X-1})for(int pm2: VI{0, 1}){
            if(d1==d2) continue;
            const auto d2_str=(pm2?"":"-")+d2+to_string(i2);
            for(const auto &d3: VS{"r","d","f"})for(int i3: VI{0, X-1})for(int pm3: VI{0, 1}){
                if(d1==d3||d2==d3) continue;
                const auto d3_str=(pm3?"":"-")+d1+to_string(i3);
                auto action=action_encode(d1_str+"."+d1_str+"."+d2_str+"."+d3_str+"."+d1_strinv);

                auto state=simulation(index, action);
                auto hash=zhash.hash(state);
                if(!ops.contains(hash) || action.size()<ops[hash].size())
                    ops[hash]=action;
            }
        }
    }
    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}

// パネルを回転させる系操作
VVI get_rotate_panel(){
    // "f5.r0"
    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);
    for(const auto& d1: VS{"r","d","f"})for(int i: VI{0, X-1})REP(sign1, 2){
        const auto a1=(sign1?"":"-")+d1+to_string(i);

        auto action=action_encode(a1);
        auto state=simulation(index, action);
        auto hash=zhash.hash(state);
        if(!ops.contains(hash) || action.size()<ops[hash].size())
            ops[hash]=action;
        
        for(const auto& d2: VS{"r","d","f"})for(int i2: VI{0, X-1})REP(sign2, 2){
            const auto a2=(sign2?"":"-")+d2+to_string(i2);
            
            action=action_encode(a1+"."+a2);
            state=simulation(index, action);
            hash=zhash.hash(state);
            if(!ops.contains(hash) || action.size()<ops[hash].size())
                ops[hash]=action;

            // for(const auto& d3: VS{"r","d","f"})for(int i3: VI{0, X-1})REP(sign3, 2){
            //     const auto a3=(sign3?"":"-")+d3+to_string(i3);

            //     action=action_encode(a1+"."+a2+"."+a3);
            //     state=simulation(index, action);
            //     hash=zhash.hash(state);
            //     if(!ops.contains(hash) || action.size()<ops[hash].size())
            //         ops[hash]=action;


            //     // for(const auto& d4: VS{"r","d","f"})for(int i4: VI{0, X-1})REP(sign4, 2){
            //     //     const auto a4=(sign4?"":"-")+d4+to_string(i4);

            //     //     action=action_encode(a1+"."+a2+"."+a3+"."+a4);
            //     //     state=simulation(index, action);
            //     //     hash=zhash.hash(state);
            //     //     if(!ops.contains(hash) || action.size()<ops[hash].size())
            //     //         ops[hash]=action;
            //     // }
            // }
        }
    }
    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}

// https://macozy.com/rubik/4rc/step_omake_etsuki_s4.html
VVI magic_center_rotate180(){
    // 180
    // -r3.r0.d3.r3.-r0.d3.d3    x2

    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);
    VVI centers=get_centers();
    for(const auto& d1: VS{"r","d","f"})for(int i: VI{0, X-1})REP(sign1, 2){
        const auto a1=(sign1?"":"-")+d1+to_string(i);
        const auto a1inv=(sign1?"-":"")+d1+to_string(i);
        for(const auto& d2: VS{"r","d","f"})for(int i2: VI{0, X-1})REP(sign2, 2){
            const auto a2=(sign2?"":"-")+d2+to_string(i2);
            const auto a2inv=(sign2?"-":"")+d2+to_string(i2);
            for(const auto& d3: VS{"r","d","f"})for(int i3: VI{0, X-1})REP(sign3, 2){
                const auto a3=(sign3?"":"-")+d3+to_string(i3);
                string tmp = a1+"."+a2+"."+a3+"."+a1inv+"."+a2inv+"."+d3+"."+d3;
                auto action=action_encode(tmp+"."+tmp);
                auto state=simulation(index, action);
                if(state==index)continue;
                auto hash=zhash.hash(state);
                bool allok=check_center_change(centers, state);
                if(allok && (!ops.contains(hash) || action.size()<ops[hash].size()))
                    ops[hash]=action;
            }
        }
    }

    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}

// https://macozy.com/rubik/4rc/step_omake_etsuki_s4.html
VVI magic_center_rotate90(){
    // 90
    // r1.r2.d3.-r1.-r2.-d3   x5

    ZobristHashing<uint64_t> zhash(state_length, state_length, rand_engine);
    unordered_map<uint64_t, VI> ops;

    VI index(state_length);
    ARANGE(index);
    VVI centers=get_centers();
    for(const auto& d1: VS{"r","d","f"})REP(sign1, 2){
        string a1="";
        FOR(i, 1, X-1)
            a1 = a1+"."+(sign1?"":"-")+d1+to_string(i);
        auto a1enc=action_encode(a1);
        auto a1invenc=inverse_action(a1enc);
        for(const auto& d2: VS{"r","d","f"})for(int i2: VI{0, X-1})REP(sign2, 2){
            const auto a2=(sign2?"":"-")+d2+to_string(i2);
            const auto a2inv=(sign2?"-":"")+d2+to_string(i2);

            auto tmp=a1enc;
            tmp.emplace_back(action_encode(a2).front());
            tmp.insert(tmp.end(), a1invenc.begin(), a1invenc.end());
            tmp.emplace_back(action_encode(a2inv).front());

            auto action=tmp;
            REP(_, 4)
                action.insert(action.end(), tmp.begin(), tmp.end());

            auto state=simulation(index, action);
            if(state==index)continue;
            auto hash=zhash.hash(state);
            bool allok=check_center_change(centers, state);
            if(allok && (!ops.contains(hash) || action.size()<ops[hash].size()))
                ops[hash]=action;
        }
    }

    VVI magics;
    for(auto&[key,value]:ops)
        magics.emplace_back(value);
    dump(SZ(magics))
    return magics;
}

optional<VI> greedy_center(const VI &init_state, int retry_num=100){
    // TODO: 順序を維持する
    auto magics=get_magic4_1();
    for(auto &m:get_magic5_1())
        magics.emplace_back(m);
    auto help_magic=get_rotate_panel();
    //
    VVI centers;
    centers.emplace_back(solution_state);
    REP(r,3){
        auto state=solution_state;
        auto &prev=centers.back();
        REP(t, 6)REP(i,X)REP(j,X){
            state[X*X*t+j*X+i]=prev[X*X*t+(X-1-i)*X+j];
        }
        centers.emplace_back(state);
    }

    auto calc_score=[&](const VI &state, int intercept){
        int score=0;
        REP(t, 6){
            int minval=INF;
            for(auto& center: centers){
                int val=0;
                FOR(i, intercept, X-intercept)FOR(j, intercept, X-intercept){
                    int idx=X*X*t+i*X+j;
                    if(state[idx]!=center[idx])val++;
                }
                minval=min(minval,val);
            }
            score+=minval;
        }
        return score;
    };
    REP(_,retry_num){
        VI result;
        auto state=init_state;
        bool fail=false;
        RFOR(intercept, 1, X/2)if(!fail){
            // intercept=1;
            dump(intercept)
            int score=calc_score(state,intercept);

            int help_num=-1;
            while(score!=0){
                shuffle(ALL(magics), rand_engine);
                bool update=false;
                dump(score)
                auto s=state;
                if(help_num>=0)
                    s=simulation(state, help_magic[help_num]);
                for(auto &magic: magics){
                    auto t=simulation(s, magic);
                    int t_score=calc_score(t, intercept);
                    if(score>t_score){
                        score=t_score;
                        state=t;
                        if(help_num>=0)
                            result.insert(result.end(), help_magic[help_num].begin(), help_magic[help_num].end());
                        result.insert(result.end(), magic.begin(), magic.end());
                        update=true;
                        break;
                    }
                }
                if(!update){
                    help_num++;
                    dump(help_num)
                    if(help_num>=SZ(help_magic)){
                        OUT("fail");
                        fail=true;
                        break;
                    }
                }else help_num=-1;
            }
            dump(score)
        }
        if(!fail)
            return result;
    }
    return nullopt;
}


optional<VI> greedy_edge(const VI &init_state, int retry_num=100){
    auto magics=get_magic4_2();
    for(auto &m:get_magic4_3())
        magics.emplace_back(m);
    for(auto &m:get_magic4_4())
        magics.emplace_back(m);
    for(auto &m:get_magic5_2())
        magics.emplace_back(m);
    
    auto help_magic=get_rotate_panel();
    for(auto &m:get_help_magic())
        help_magic.emplace_back(m);

    VVI edges;
    REP(t, 6){
        for(int i: VI{0, X-1}){
            VI edge;
            REP(j, X)
                edge.emplace_back(solution_state[X*X*t+i*X+j]);
            edges.emplace_back(edge);
            REVERSE(edge);
            edges.emplace_back(edge);
        }
        for(int j: VI{0, X-1}){
            VI edge;
            REP(i, X)
                edge.emplace_back(solution_state[X*X*t+i*X+j]);
            edges.emplace_back(edge);
            REVERSE(edge);
            edges.emplace_back(edge);
        }
    }

    auto calc_score=[&](const VI &state, int intercept){
        int score=0;
        REP(t, 6){
            for(int i: VI{0, X-1}){
                int minval=INF;
                for(auto& edge: edges){
                    int val=0;
                    FOR(j, intercept, X-intercept){
                        if(state[X*X*t+i*X+j]!=edge[j])val++;
                    }
                    minval=min(minval,val);
                }
                score+=minval;
            }
            for(int j: VI{0, X-1}){
                int minval=INF;
                for(auto& edge: edges){
                    int val=0;
                    FOR(i, intercept, X-intercept){
                        if(state[X*X*t+i*X+j]!=edge[i])val++;
                    }
                    minval=min(minval,val);
                }
                score+=minval;
            }
        }
        return score;
    };
    VI result_all;
    auto state=init_state;
    int score=INF;
    RFOR(intercept, 1, X/2){
        dump(intercept)
        auto save_state=state;
        REP(retry, retry_num){
            auto state=save_state;
            score=calc_score(state, intercept);
            int help_num=-1;
            VI result;
            while(score!=0){
                shuffle(ALL(magics), rand_engine);
                bool update=false;
                OUT("intercept", intercept, "retry", retry, "score", score, "help_num", help_num);
                auto s=state;
                if(help_num>=0)
                    s=simulation(state, help_magic[help_num]);
                for(auto &magic: magics){
                    auto t=simulation(s, magic);
                    int t_score=calc_score(t, intercept);
                    if(score>t_score){
                        score=t_score;
                        state=t;
                        if(help_num>=0)
                            result.insert(result.end(), help_magic[help_num].begin(), help_magic[help_num].end());
                        result.insert(result.end(), magic.begin(), magic.end());
                        update=true;
                        break;
                    }
                }
                if(!update){
                    help_num++;
                    if(help_num>=SZ(help_magic)){
                        OUT("fail");
                        dump(action_decode(result))
                        break;
                    }
                }else help_num=-1;
            }
            if(score==0){
                result_all.insert(result_all.end(), ALL(result));
                break;
            }
            dump(score)
        }
    }
    if(score==0)
        return result_all;
    return nullopt;
}

optional<VI> last_center(const VI &init_state, int retry_num=10000){
    auto magics=get_magic5_1();
    // for(auto &m :magic_center_rotate90())
    //     magics.emplace_back(m);
    // for(auto &m :magic_center_rotate180())
    //     magics.emplace_back(m);
    // auto help_magic=get_rotate_panel();
    auto help_magic=get_magic5_1();

    auto calc_score=[&](const VI &state){
        int score=0;
        REP(t, 6){
            int val=0;
            REP(i, X)REP(j, X){
                int idx=X*X*t+i*X+j;
                if(state[idx]!=solution_state[idx])val++;
            }
            score+=val;
        }
        return score;
    };
    REP(_,retry_num){
        VI result;
        auto state=init_state;
        bool fail=false;
        int score=calc_score(state);

        int help_num=-1;
        while(score!=0){
            shuffle(ALL(magics), rand_engine);
            bool update=false;
            dump(score)
            auto s=state;
            if(help_num>=0)
                s=simulation(state, help_magic[help_num]);
            for(auto &magic: magics){
                auto t=simulation(s, magic);
                int t_score=calc_score(t);
                if(score>t_score){
                    score=t_score;
                    state=t;
                    if(help_num>=0)
                        result.insert(result.end(), help_magic[help_num].begin(), help_magic[help_num].end());
                    result.insert(result.end(), magic.begin(), magic.end());
                    update=true;
                    break;
                }
            }
            if(!update){
                help_num++;
                dump(help_num)
                if(help_num>=SZ(help_magic)){
                    OUT("fail");
                    fail=true;
                    break;
                }
            }else help_num=-1;
        }
        dump(score)
        if(!fail)
            return result;
    }
    return nullopt;
}


optional<VI> center_rotation(const VI &init_state, int retry_num=10000){
    auto magics=magic_center_rotate90();
    for(auto &m :magic_center_rotate180())
        magics.emplace_back(m);
    auto help_magic=magic_center_rotate90();

    auto calc_score=[&](const VI &state){
        int score=0;
        REP(t, 6){
            int val=0;
            REP(i, X)REP(j, X){
                int idx=X*X*t+i*X+j;
                if(state[idx]!=solution_state[idx])val++;
            }
            score+=val;
        }
        return score;
    };
    REP(_,retry_num){
        VI result;
        auto state=init_state;
        bool fail=false;
        int score=calc_score(state);

        int help_num=-1;
        while(score!=0){
            shuffle(ALL(magics), rand_engine);
            bool update=false;
            dump(score)
            auto s=state;
            if(help_num>=0)
                s=simulation(state, help_magic[help_num]);
            for(auto &magic: magics){
                auto t=simulation(s, magic);
                int t_score=calc_score(t);
                if(score>t_score){
                    score=t_score;
                    state=t;
                    if(help_num>=0)
                        result.insert(result.end(), help_magic[help_num].begin(), help_magic[help_num].end());
                    result.insert(result.end(), magic.begin(), magic.end());
                    update=true;
                    break;
                }
            }
            if(!update){
                help_num++;
                dump(help_num)
                if(help_num>=SZ(help_magic)){
                    OUT("fail");
                    fail=true;
                    break;
                }
            }else help_num=-1;
        }
        dump(score)
        if(!fail)
            return result;
    }
    return nullopt;
}


const string PROBLEM_DATA_DIR = "./data/";
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
    //    {"wreath_6/6", 12},
    //    {"wreath_7/7", 12},
    //    {"wreath_12/12", 12},
    //    {"wreath_21/21", 12},
    //    {"wreath_33/33", 12},
    //    {"wreath_100/100", 9},
    //    {"globe_1/8", 4},
    //    {"globe_1/16", 3},
    //    {"globe_2/6", 4},
    //    {"globe_3/4", 4},
    //    {"globe_6/4", 4},
    //    {"globe_6/8", 3},
    //    {"globe_6/10", 3},
    //    {"globe_3/33", 2},
    //    {"globe_8/25", 2}
};

VI greedy_solver(){
    auto result1=greedy_center(initial_state);
    assert(result1.has_value());
    dump(action_decode(result1.value()))

    auto state=simulation(initial_state, result1.value());
    auto result2=greedy_edge(state);
    dump(action_decode(result2.value()))

    VI path=result1.value();
    path.insert(path.end(), result2.value().begin(), result2.value().end());
    dump(action_decode(path))
    return path;
}

int main() {
    ChronoTimer timer;
    int case_num = 398;
    double sum_score = 0.0;
    double sum_log_score = 0.0;
    int64_t max_time = 0;
    // REP(i, case_num){
    // for(int i: VI{240}){
    for(int i: VI{283}){
    // FOR(i, 255, 256+1){
    // FOR(i, 245, 256+1){
        timer.start();
        dump(SEED)
        rand_engine.seed(SEED);
        string input_filename = to_string(i) + ".txt";
        string file_path = PROBLEM_DATA_DIR + input_filename;
        ifstream ifs(file_path);
        assert(!ifs.fail());
        data_load(ifs);
        if(!TARGET.contains(puzzle_type))
            continue;
        OUT("id", i);
        // if(num_wildcards==0)continue;
        double score=0;

        string output_filename="output/"+to_string(i)+".txt";

        // ARANGE(solution_state);
        // initial_state=simulation_inverse(solution_state, load_actions(output_filename));
        // // ARANGE(initial_state);
        // // solution_state=simulation(initial_state, load_actions(output_filename));
        // label_mapping.clear();
        // REP(i, state_length)
        //     label_mapping["N"+to_string(i)]=i;
        // greedy_solver();
    
        // auto res=last_center(simulation(initial_state, action_encode("d0.d0.r0.r1.r0.r1.f0.f1.f0.f1.-d4.-d4.-d4.-d3.f0.f1.-d0.-d1.f4.d4.d3.-d0.-d1.-f0.-f1.d0.f0.f1.f0.f1.d0.d1.d0.d1.-r0.-r1.-f4.-f0.r4.r3.d4.f0.f1.f0.f1.r4.r3.f0.f1.f0.f1.d0.d0.f0.d4.-r0.d0.d1.d0.d1.-r4.-r3.-r4.-r3.-d4.-r0.f0.d0.-r4.-r3.-r4.-r3.-d4.-d3.-d4.-d3.-r4.-r3.-r4.-r3.-d4.-d3.-d4.-d3.d0.d0.-f0.d0.d0.-r4.-r4.d0.d1.d0.d1.-f4.-r4.-r4.-d4.-d3.-d4.-d3.-r4.-r3.-r4.-r3.-f4.d0.r0.r1.r0.r1.d0.d0.-r4.-r4.-f4.-f3.-f4.-f3.-r4.-r3.-r4.-r3.f0.f0.r0.r0.d4.-r4.-r3.-r4.-r3.-d4.-d4.r0.r1.r0.r1.-d0.-f4.-f4.-r4.-r3.-r4.-r3.f0.f1.f0.f1.d4.d0.r4.f0.f0.d0.f4.f0.r4.r0.-d4.f4.f0.f0.-d0.f0.f0.-d4.-f4.-f4.-d4.-d4.-r4.-r4.d4.f0.f0.-d0.d4.d4.d3.d3.d2.d2.d1.d1.d0.d0")));
        // assert(res);
        // dump(action_decode(res.value()))
        auto state=simulation(initial_state, load_actions(to_string(i)+".txt"));
        auto res=center_rotation(state);
        assert(res);
        dump(action_decode(res.value()))
        return 0;

        // score = cube_solver();
        // return 0;

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
