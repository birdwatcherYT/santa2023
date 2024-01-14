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


VI solve_center(){
    int size=X*X;
    auto center_penalty = [&](int panel, const VI& state){
        int penalty=0;
        FOR(i, 1, X-1)FOR(j, 1, X-1){
            int idx=size*panel+i*X+j;
            penalty+=state[idx]!=solution_state[idx];
        }
        if(penalty%2==1)
            penalty++;
        penalty/=2;
        return penalty;
        // int row_penalty=0;
        // FOR(i, 1, X-1){
        //     int row=0;
        //     FOR(j, 1, X-1){
        //         int idx=size*panel+i*X+j;
        //         row+=state[idx]!=solution_state[idx];
        //     }
        //     row_penalty+=(row!=0);
        // }
        // int col_penalty=0;
        // FOR(j, 1, X-1){
        //     int col=0;
        //     FOR(i, 1, X-1){
        //         int idx=size*panel+i*X+j;
        //         col+=state[idx]!=solution_state[idx];
        //     }
        //     col_penalty+=(col!=0);
        // }
        // return max(row_penalty, col_penalty);
    };

    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);

    VI target_panel;
    SI target_label;
    auto center_penalty_cumsum = [&](const VI& state){
        int penalty=0;
        for(int t: target_panel)
            penalty+=center_penalty(t, state);
        return penalty;
    };

    function<bool(const VI &, VI &, int, int, unordered_set<uint64_t> &)> search;
    search=[&](const VI &state, VI &path, int same_action_num, int max_action_size, unordered_set<uint64_t> &visit){
        int h=center_penalty_cumsum(state);
        if (h==0)
            return true;
        // if(SZ(path) >= max_action_size) return false;
        if(h+SZ(path) >= max_action_size) return false;
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
            auto next_hash=zhash.hash(next, target_label);
            if (visit.contains(next_hash))
                continue;
            visit.emplace(next_hash);
            path.emplace_back(a);
            if (search(next, path, next_same_action_num, max_action_size, visit))
                return true;
            path.pop_back();
            visit.erase(next_hash);
        }
        return false;
    };


    auto state=initial_state;
    VI all_path;
    VI panels{0,5, 1,2, 4,3};
    shuffle(ALL(panels), rand_engine);
    // VI panels{0,5, 1,3, 2,4};
    // for(auto [panel1,panel2]: vector<PII>{{0,5}, {1,3}, {2,4}}){
    for(auto panel: panels){
        target_panel.emplace_back(panel);
        // target_panel.emplace_back(panel1);
        // target_panel.emplace_back(panel2);
        
        // for(int panel: VI{panel1, panel2})
        {
            FOR(i, 1, X-1)FOR(j, 1, X-1){
                int idx=size*panel+i*X+j;
                target_label.emplace(solution_state[idx]);
            }
        }

        FOR(max_action_size, 1, INF){
            dump(max_action_size)
            unordered_set<uint64_t> visit;
            visit.emplace(zhash.hash(state, target_label));
            VI path;
            if(search(state, path, 0, max_action_size, visit)){
                OUT("Find!", path, target_panel);
                state=simulation(state, path);
                all_path.insert(all_path.end(), path.begin(), path.end());
                break;
            }
        }
        // // hash -> {prev_hash, action_id}
        // unordered_map<uint64_t, tuple<uint64_t,int>> pushed;
        // // mistake, length, hash
        // MINPQ<tuple<int, int, uint64_t>> pq;
        // auto init_hash = zhash.hash(state, target_label);
        // pushed[init_hash]={0, -1};
        // int p=center_penalty_cumsum(state);
        // pq.emplace(p, p, init_hash);
        // int searched=0;
        // while(!pq.empty()){
        //     auto [_, mistake, hash] = pq.top(); pq.pop();
        //     searched++;
        //     if(searched%100000==0)
        //         dump(searched)
        //     auto actions=construct_actions(init_hash, hash, pushed);
        //     auto current_state=simulation(state, actions);
        //     if(mistake==0){
        //         // OUT("find", panel1, panel2, searched);
        //         OUT("find", panel, searched);
        //         state=current_state;
        //         break;
        //     }
        //     REP(a, allowed_action_num*2){
        //         auto next = do_action(current_state, a);
        //         auto next_hash = zhash.hash(next);
        //         if(pushed.contains(next_hash))
        //             continue;
        //         pushed[next_hash]={hash, a};
        //         p=center_penalty_cumsum(next);
        //         pq.emplace(p+SZ(actions)+1, p, next_hash);
        //         if(pq.size()%5000000==0)
        //             dump(pq.size())
        //         if(pushed.size()%5000000==0)
        //             dump(pushed.size())
        //     }
        // }
    }
    return all_path;
}

VI solve_edge(const VI& start){
    int size=X*X;
    auto center_penalty = [&](int panel, const VI& state){
        int penalty=0;
        FOR(i, 1, X-1)FOR(j, 1, X-1){
            int idx=size*panel+i*X+j;
            penalty+=state[idx]!=solution_state[idx];
        }
        if(penalty%2==1)
            penalty++;
        penalty/=2;
        return penalty;
    };
    map<VI, int> edges;
    REP(panel, 6)for(int j: VI{0, X-1}){
        VI e1,e2;
        FOR(i, 1, X-1){
            e1.emplace_back(solution_state[size*panel+i*X+j]);
            e2.emplace_back(solution_state[size*panel+j*X+i]);
        }
        edges[e1]++;
        edges[e2]++;
        // edges.emplace_back(e1);
        // edges.emplace_back(e2);
    }

    auto edge_ok = [&](int panel, const VI& state){
        int ok=0;
        for(int j: VI{0, X-1}){
            VI e1,e2;
            FOR(i, 1, X-1){
                e1.emplace_back(state[size*panel+i*X+j]);
                e2.emplace_back(state[size*panel+j*X+i]);
            }
            if(edges.contains(e1))
                ok++;
            if(edges.contains(e2))
                ok++;
        }
        return ok;
    };

    ZobristHashing<uint64_t> zhash(SZ(label_mapping), state_length, rand_engine);

    auto center_penalty_cumsum = [&](const VI& state, int edge_num){
        int penalty=0;
        int ok_e=0;
        REP(t, 6){
            penalty+=center_penalty(t, state);
            ok_e+=edge_ok(t, state);
        }
        return penalty+max(0, edge_num-ok_e);
    };

    function<bool(const VI &, VI &, int, int, unordered_set<uint64_t> &, int)> search;
    search=[&](const VI &state, VI &path, int same_action_num, int max_action_size, unordered_set<uint64_t> &visit, int edge_num){
        int h=center_penalty_cumsum(state, edge_num);
        if (h==0)
            return true;
        // if(SZ(path) >= max_action_size) return false;
        if(h+SZ(path) >= max_action_size) return false;
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
            if (search(next, path, next_same_action_num, max_action_size, visit, edge_num))
                return true;
            path.pop_back();
            visit.erase(next_hash);
        }
        return false;
    };

    auto state=start;
    VI all_path;
    FOR(i, 1, 6*4+1){
        FOR(max_action_size, 1, INF){
            dump(max_action_size)
            unordered_set<uint64_t> visit;
            visit.emplace(zhash.hash(state));
            VI path;
            if(search(state, path, 0, max_action_size, visit, i)){
                OUT("Find!", path, i);
                state=simulation(state, path);
                all_path.insert(all_path.end(), path.begin(), path.end());
                break;
            }
        }
    }
    return all_path;
}

int cube_solver(){
    auto center_path=solve_center();
    dump(center_path);
    dump(action_decode(center_path));

    // 200
    // VI center_path = { 0, 22, 2, 23, 0, 22, 15, 22, 6, 0, 0, 16, 18, 7, 7, 21, 2, 19, 2, 16, 2, 16, 2, 4, 2, 7, 14, 3, 9, 7, 21, 4, 15, 16, };
    // // // action_decode(center_path) = f0.-d2.f2.-d3.f0.-d2.-f3.-d2.r2.f0.f0.-r0.-r2.r3.r3.-d1.f2.-r3.f2.-r0.f2.-r0.f2.r0.f2.r3.-f2.f3.d1.r3.-d1.r0.-f3.-r0
    // VI edge_path{ 12, 8, 0, 20, 15, 20, 3, 8, 3, 16, 15, 4, 5, 11, 17, 20, 14, 8, 2, 8, 1, 3, 4, 15, 16, 13, 4, 17, 12, 16, 0, 5, 11, 16, 23, 4, 3, 18, 12, 6, 0, 3, 3, 8, 3, 20, 12, 11, 0, 23, 12, 12, 8, 0, 0, 20, 12, 12, 20, 0, 14, 8, 0, 20, 2, 8, 12, 0, 11, 14, 11, 12, 23, 2, 3, 23, 15, 0, 23, 12, 11, 3, 16, 14, 4, 15, 16, 2, 4, 15, 11, 3, 23, };
    // action_decode(edge_path) = -f0.d0.f0.-d0.-f3.-d0.f3.d0.f3.-r0.-f3.r0.r1.d3.-r1.-d0.-f2.d0.f2.d0.f1.f3.r0.-f3.-r0.-f1.r0.-r1.-f0.-r0.f0.r1.d3.-r0.-d3.r0.f3.-r2.-f0.r2.f0.f3.f3.d0.f3.-d0.-f0.d3.f0.-d3.-f0.-f0.d0.f0.f0.-d0.-f0.-f0.-d0.f0.-f2.d0.f0.-d0.f2.d0.-f0.f0.d3.-f2.d3.-f0.-d3.f2.f3.-d3.-f3.f0.-d3.-f0.d3.f3.-r0.-f2.r0.-f3.-r0.f2.r0.-f3.d3.f3.-d3

    // 200
    // VI center_path{ 15, 10, 10, 15, 2, 8, 13, 10, 4, 5, 3, 17, 22, 4, 4, 11, 6, 11, 14, 7, 2, 18, 0, 11, 0, 19, 9, 12, 21, 0, 0, 7, 12, };
    // action_decode(center_path) = -f3.d2.d2.-f3.f2.d0.-f1.d2.r0.r1.f3.-r1.-d2.r0.r0.d3.r2.d3.-f2.r3.f2.-r2.f0.d3.f0.-r3.d1.-f0.-d1.f0.f0.r3.-f0

    // 205
    // // Find! [ 1, 16, 1, 4, 13, 16, 13, 23, 1, 4, 11, 13, ] [ 3, 2, 1, 5, 4, 0, ]
    // VI center_path{ 2, 4, 13, 5, 6, 1, 1, 2, 16, 10, 4, 22, 0, 8, 10, 0, 22, 5, 0, 17, 1, 7, 1, 8, 8, 23, 13, 19, 13, 2, 11, 14, 7, 1, 11, 13, 1, 16, 1, 4, 13, 16, 13, 23, 1, 4, 11, 13, };
    // // action_decode(center_path) = f2.r0.-f1.r1.r2.f1.f1.f2.-r0.d2.r0.-d2.f0.d0.d2.f0.-d2.r1.f0.-r1.f1.r3.f1.d0.d0.-d3.-f1.-r3.-f1.f2.d3.-f2.r3.f1.d3.-f1.f1.-r0.f1.r0.-f1.-r0.-f1.-d3.f1.r0.d3.-f1
    // VI edge_path{ 6, 3, 8, 15, 20, 18, 4, 20, 16, 8, 21, 16, 0, 4, 12, 9, 4, 11, 16, 23, 10, 3, 16, 15, 4, 22, 0, 8, 12, 20, 8, 15, 20, 19, 3, 7, 9, 0, 4, 12, 16, 21, 11, 16, 9, 19, 23, 7, 21, 0, 4, 12, 23, 16, 11, 4, 9, 4, 0, 16, 12, 21, 10, 10, 19, 0, 7, 12, 10, 10, 12, 23, 0, 22, 0, 11, 12, 10, 4, 3, 16, 15, 10, 4, 23, 16, 9, 11, 12, 21, 22, 0, 19, 8, 12, 20, 3, 19, 15, 8, 7, 20, 8, 7, 20, 19, };
    // // action_decode(edge_path) = r2.f3.d0.-f3.-d0.-r2.r0.-d0.-r0.d0.-d1.-r0.f0.r0.-f0.d1.r0.d3.-r0.-d3.d2.f3.-r0.-f3.r0.-d2.f0.d0.-f0.-d0.d0.-f3.-d0.-r3.f3.r3.d1.f0.r0.-f0.-r0.-d1.d3.-r0.d1.-r3.-d3.r3.-d1.f0.r0.-f0.-d3.-r0.d3.r0.d1.r0.f0.-r0.-f0.-d1.d2.d2.-r3.f0.r3.-f0.d2.d2.-f0.-d3.f0.-d2.f0.d3.-f0.d2.r0.f3.-r0.-f3.d2.r0.-d3.-r0.d1.d3.-f0.-d1.-d2.f0.-r3.d0.-f0.-d0.f3.-r3.-f3.d0.r3.-d0.d0.r3.-d0.-r3

    auto state=simulation(initial_state, center_path);
    auto edge_path=solve_edge(state);
    dump(edge_path);
    dump(action_decode(edge_path));
    return INF;
}

const string DATA_DIR = "./data/";
map<string, int> TARGET{
    //    {"cube_2/2/2", 5},
    //    {"cube_3/3/3", 4},
       {"cube_4/4/4", 4},
    //    {"cube_5/5/5", 3},
    //    {"cube_6/6/6", 3},
    //    {"cube_7/7/7", 3},
    //    {"cube_8/8/8", 3},
    //    {"cube_9/9/9", 3},
    //    {"cube_10/10/10", 3},
    //    {"cube_19/19/19", 2},
    //    {"cube_33/33/33", 1},
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

int main() {
    ChronoTimer timer;
    int case_num = 398;
    double sum_score = 0.0;
    double sum_log_score = 0.0;
    int64_t max_time = 0;
    // REP(i, case_num){
    // FOR(i, 205, 210){
    FOR(i, 205+1, 210){
    // FOR(i, 200, 210){
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

        score = cube_solver();

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
