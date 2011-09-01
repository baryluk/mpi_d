module mpi;

enum MPI_VERSION = 1;
enum MPI_SUBVERSION = 2;

enum MPI_Datatype {
	MPI_CHAR, /// char
	MPI_BYTE, /// See standard; like unsigned char
	MPI_SHORT, /// short
	MPI_INT, /// int
	MPI_LONG, /// long
	MPI_FLOAT, /// float
	MPI_DOUBLE, /// double
	MPI_UNSIGNED_CHAR, /// unsigned char
	MPI_UNSIGNED_SHORT, /// unsigned short
	MPI_UNSIGNED, /// unsigned int
	MPI_UNSIGNED_LONG, /// unsigned long
	MPI_LONG_DOUBLE, /// long double (some systems may not implement)
	MPI_LONG_LONG_INT, /// long long (some systems may not implement)

	// The following are datatypes for the MPI functions MPI_MAXLOC and MPI_MINLOC.
	MPI_FLOAT_INT, /// struct { float, int }
	MPI_LONG_INT, /// struct { long, int }
	MPI_DOUBLE_INT, /// struct { double, int }
	MPI_SHORT_INT, /// struct { short, int }
	MPI_2INT, /// struct { int, int }
	MPI_LONG_DOUBLE_INT,  /// struct { long double, int }; this is an OPTIONAL type, and may be set to NULL

	/// Special datatypes for C and Fortran
	MPI_PACKED,
	MPI_UB,
	MPI_LB
}

/// TODO: What real types this haves? Currently this is just to silent temporarly compiler. */
alias int MPI_Comm;
/// ditto
alias int MPI_Errhandler;
/// ditto
alias int MPI_Group;
/// ditto
alias int MPI_Request;
/// ditto
alias int MPI_Aint;

/// Contains all of the processes
extern MPI_Comm MPI_COMM_WORLD;
/// Contains only the calling process
extern MPI_Comm MPI_COMM_SELF;

/// A group containing no members.
extern MPI_Group MPI_GROUP_EMPTY;

/// Results of the compare operations
enum {
	MPI_IDENT = 1, /// Identical
	MPI_CONGRUENT = 2, /// (only for MPI_COMM_COMPARE) The groups are identical
	MPI_SIMILAR = 3, /// Same members, but in a different order
	MPI_UNEQUAL = 4 /// Different
}

/// Collective operations
/* The collective combination operations (MPI_REDUCE, MPI_ALLREDUCE, MPI_REDUCE_SCATTER, and MPI_SCAN)
take a combination operation. This operation is of type MPI_Op in C and of type INTEGER in Fortran.
The predefined operations are 
*/
enum MPI_Op {
	MPI_MAX, /// return the maximum
	MPI_MIN, /// return the minumum
	MPI_SUM, /// return the sum
	MPI_PROD, /// return the product
	MPI_LAND, /// return the logical and
	MPI_BAND, /// return the bitwise and
	MPI_LOR, /// return the logical or
	MPI_BOR, /// return the bitwise of
	MPI_LXOR, /// return the logical exclusive or
	MPI_BXOR, /// return the bitwise exclusive or
	MPI_MINLOC, /// return the minimum and the location (actually, the value of the second element of the structure where the minimum of the first is found)
	MPI_MAXLOC /// return the maximum and the location
}

enum {
	MPI_TAG_UB = 0, /// Largest tag value
	MPI_HOST = 1, /// Rank of process that is host, if any
	MPI_IO = 2, /// Rank of process that can do I/O
	MPI_WTIME_IS_GLOBAL = 3 /// Has value 1 if MPI_WTIME is globally synchronized.
}

/// Topology types
enum {
	MPI_GRAPH = 2, /// General graph
	MPI_CART = 1 /// Cartesian grid
}

/** TOOD

MPI_SOURCE, /// Who sent the message
MPI_TAG, /// What tag the message was sent with
MPI_ERROR /// Any error return
*/

enum MPI_ANY_TAG = -1;
enum MPI_ANY_SOURCE = -1;

/// LAM
enum MPI_Status {
	MPI_SUCCESS = 0, /// Successful return code
	MPI_ERR_BUFFER = 1, /// Invalid buffer pointer
	MPI_ERR_COUNT = 2, /// Invalid count argument
	MPI_ERR_TYPE = 3, /// Invalid datatype argument
	MPI_ERR_TAG = 4, /// Invalid tag argument
	MPI_ERR_COMM = 5, /// Invalid communicator
	MPI_ERR_RANK = 6, /// Invalid rank
	MPI_ERR_ROOT = 8, /// Invalid root
	MPI_ERR_GROUP = 9, /// Null group passed to function
	MPI_ERR_OP = 10, /// Invalid operation
	MPI_ERR_TOPOLOGY = 11, /// Invalid topology
	MPI_ERR_DIMS = 12, /// Illegal dimension argument
	MPI_ERR_ARG = 13, /// Invalid argument
	MPI_ERR_UNKNOWN = 14, /// Unknown error
	MPI_ERR_TRUNCATE = 15, /// message truncated on receive
	MPI_ERR_OTHER = 16, /// Other error; use Error_string
	MPI_ERR_INTERN = 17, /// internal error code
	MPI_ERR_IN_STATUS = 18, /// Look in status for error value
	MPI_ERR_PENDING = 19, /// Pending request
	MPI_ERR_REQUEST = 7, /// illegal mpi_request handle
	MPI_ERR_LASTCODE = 37 /// Last error code -- always at end
}

/// 3.2.1 Blocking send
extern(C) int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
/// 3.2.4 Blocking receive
extern(C) int MPI_Recv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);

/// 3.2.5 Return status
extern(C) int MPI_Get_count(MPI_Status *status, MPI_Datatype datatype, int *count);

/// 3.4 Communication Modes
extern(C) int MPI_Bsend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
extern(C) int MPI_Ssend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
extern(C) int MPI_Rsend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);

/// 3.6 Buffer allocation and usage
extern(C) int MPI_Buffer_attach(void* buffer, int size);
extern(C) int MPI_Buffer_detach(void* buffer_addr, int* size);

/// 3.7.2 Communication initiation
extern(C) int MPI_Isend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
extern(C) int MPI_Ibsend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
extern(C) int MPI_Issend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
extern(C) int MPI_Irsend(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
extern(C) int MPI_Irecv(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);

/// 3.7.3. Communication Completion
extern(C) int MPI_Wait(MPI_Request *request, MPI_Status  *status);
extern(C) int MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
extern(C) int MPI_Request_free(MPI_Request *request);

/// 3.7.5. Multiple Completions
extern(C) int MPI_Waitany(int count, MPI_Request *array_of_requests, int *index, MPI_Status *status);
extern(C) int MPI_Testany(int count, MPI_Request *array_of_requests, int *index, int *flag, MPI_Status *status);
extern(C) int MPI_Waitall(int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses);
extern(C) int MPI_Testall(int count, MPI_Request *array_of_requests, int *flag, MPI_Status *array_of_statuses);
extern(C) int MPI_Waitsome(int incount, MPI_Request *array_of_requests, int *outcount, int *array_of_indices, MPI_Status *array_of_statuses);
extern(C) int MPI_Testsome(int incount, MPI_Request *array_of_requests, int *outcount, int *array_of_indices, MPI_Status *array_of_statuses);

/// 3.8. Probe and Cancel
extern(C) int MPI_Iprobe(int source, int tag, MPI_Comm comm, int *flag, MPI_Status *status);
extern(C) int MPI_Probe(int source, int tag, MPI_Comm comm, MPI_Status *status);
extern(C) int MPI_Test_cancelled(MPI_Status *status, int *flag);

/// 3.9. Persistent communication requests
extern(C) int MPI_Send_init(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
extern(C) int MPI_Bsend_init(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
extern(C) int MPI_Ssend_init(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
extern(C) int MPI_Rsend_init(void* buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
extern(C) int MPI_Recv_init(void* buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
extern(C) int MPI_Start(MPI_Request *request);
extern(C) int MPI_Startall(int count, MPI_Request *array_of_requests);

/// 3.10. Send-receive
// We use the same naming conventions as for blocking communication: a prefix of B, S, or R is used for buffered, synchronous or ready mode. In addition a prefix of I (for immediate) indicates that the call is nonblocking.

extern(C) int MPI_Sendrecv(void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
extern(C) int MPI_Sendrecv_replace(void* buf, int count, MPI_Datatype datatype, int dest, int sendtag, int source, int recvtag, MPI_Comm comm, MPI_Status *status);


/// 3.12.1 Datatype constructors

extern(C) int MPI_Type_contiguous(int count, MPI_Datatype oldtype, MPI_Datatype *newtype);
extern(C) int MPI_Type_vector(int count, int blocklength, int stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
extern(C) int MPI_Type_hvector(int count, int blocklength, MPI_Aint stride, MPI_Datatype oldtype, MPI_Datatype *newtype);
extern(C) int MPI_Type_indexed(int count, int *array_of_blocklengths, int *array_of_displacements, MPI_Datatype oldtype, MPI_Datatype *newtype);
extern(C) int MPI_Type_hindexed(int count, int *array_of_blocklengths, MPI_Aint *array_of_displacements, MPI_Datatype oldtype, MPI_Datatype *newtype);
extern(C) int MPI_Type_struct(int count, int *array_of_blocklengths, MPI_Aint *array_of_displacements, MPI_Datatype *array_of_types, MPI_Datatype *newtype);

extern(C) int MPI_Address(void* location, MPI_Aint *address);
extern(C) int MPI_Type_extent(MPI_Datatype datatype, MPI_Aint *extent);
extern(C) int MPI_Type_size(MPI_Datatype datatype, int *size);

extern(C) int MPI_Type_lb(MPI_Datatype datatype, MPI_Aint* displacement);
extern(C) int MPI_Type_ub(MPI_Datatype datatype, MPI_Aint* displacement);

extern(C) int MPI_Type_commit(MPI_Datatype *datatype);
extern(C) int MPI_Type_free(MPI_Datatype *datatype);

extern(C) int MPI_Get_elements(MPI_Status *status, MPI_Datatype datatype, int *count);

extern(C) int MPI_Pack(void* inbuf, int incount, MPI_Datatype datatype, void *outbuf, int outsize, int *position, MPI_Comm comm);
extern(C) int MPI_Unpack(void* inbuf, int insize, int *position, void *outbuf, int outcount, MPI_Datatype datatype, MPI_Comm comm);
extern(C) int MPI_Pack_size(int incount, MPI_Datatype datatype, MPI_Comm comm, int *size);

/// 4 Collective Communication
extern(C) int MPI_Barrier(MPI_Comm comm);

extern(C) int MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm);

extern(C) int MPI_Gather(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
extern(C) int MPI_Gatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, int root, MPI_Comm comm);
extern(C) int MPI_Scatter(void *sendbuf, int sendcnt, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, int root,  MPI_Comm comm);
extern(C) int MPI_Scatterv(void *sendbuf, int *sendcnts, int *displs, MPI_Datatype sendtype, void *recvbuf, int recvcnt, MPI_Datatype recvtype, int root, MPI_Comm comm);
extern(C) int MPI_Allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
extern(C) int MPI_Allgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *displs, MPI_Datatype recvtype, MPI_Comm comm);
extern(C) int MPI_Alltoall(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
extern(C) int MPI_Alltoallv(void* sendbuf, int *sendcounts, int *sdispls, MPI_Datatype sendtype, void* recvbuf, int *recvcounts, int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

extern(C) int MPI_Reduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

alias extern(C) void function( void *invec, void *inoutvec, int *len, MPI_Datatype *datatype) MPI_User_function;
extern(C) int MPI_Op_create(MPI_User_function *f, int commute, MPI_Op *op);

extern(C) int MPI_Allreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
extern(C) int MPI_Reduce_scatter(void* sendbuf, void* recvbuf, int *recvcounts, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
extern(C) int MPI_Scan(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

/// 5. Groups, Contexts, and Communicators
extern(C) int MPI_Group_size(MPI_Group group, int *size);
extern(C) int MPI_Group_rank(MPI_Group group, int *rank);
extern(C) int MPI_Group_translate_ranks (MPI_Group group1, int n, int *ranks1, MPI_Group group2, int *ranks2);
extern(C) int MPI_Group_compare(MPI_Group group1,MPI_Group group2, int *result);
extern(C) int MPI_Comm_group(MPI_Comm comm, MPI_Group *group);
extern(C) int MPI_Group_union(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
extern(C) int MPI_Group_intersection(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
extern(C) int MPI_Group_difference(MPI_Group group1, MPI_Group group2, MPI_Group *newgroup);
extern(C) int MPI_Group_incl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
extern(C) int MPI_Group_excl(MPI_Group group, int n, int *ranks, MPI_Group *newgroup);
extern(C) int MPI_Group_range_incl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
extern(C) int MPI_Group_range_excl(MPI_Group group, int n, int ranges[][3], MPI_Group *newgroup);
extern(C) int MPI_Group_free(MPI_Group *group);

extern(C) int MPI_Comm_size(MPI_Comm comm, int *size);
extern(C) int MPI_Comm_rank(MPI_Comm comm, int *rank);
extern(C) int MPI_Comm_compare(MPI_Comm comm1,MPI_Comm comm2, int *result);
extern(C) int MPI_Comm_dup(MPI_Comm comm, MPI_Comm *newcomm);
extern(C) int MPI_Comm_create(MPI_Comm comm, MPI_Group group, MPI_Comm *newcomm);
extern(C) int MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
extern(C) int MPI_Comm_free(MPI_Comm *comm);

extern(C) int MPI_Comm_test_inter(MPI_Comm comm, int *flag);
extern(C) int MPI_Comm_remote_size(MPI_Comm comm, int *size);
extern(C) int MPI_Comm_remote_group(MPI_Comm comm, MPI_Group *group);
extern(C) int MPI_Intercomm_create(MPI_Comm local_comm, int local_leader, MPI_Comm peer_comm, int remote_leader, int tag, MPI_Comm *newintercomm);
extern(C) int MPI_Intercomm_merge(MPI_Comm intercomm, int high, MPI_Comm *newintracomm);

/// 5.7 Caching
extern(C) int MPI_Keyval_create(MPI_Copy_function *copy_fn, MPI_Delete_function *delete_fn, int *keyval, void* extra_state);
alias extern(C) int function(MPI_Comm oldcomm, int keyval, void *extra_state, void *attribute_val_in, void *attribute_val_out, int *flag) MPI_Copy_function;
alias extern(C) int function(MPI_Comm comm, int keyval, void *attribute_val, void *extra_state) MPI_Delete_function;
extern(C) int MPI_Keyval_free(int *keyval);
extern(C) int MPI_Attr_put(MPI_Comm comm, int keyval, void* attribute_val);
extern(C) int MPI_Attr_get(MPI_Comm comm, int keyval, void *attribute_val, int *flag);
extern(C) int MPI_Attr_delete(MPI_Comm comm, int keyval);

/// 6. Process Topologies
extern(C) int MPI_Cart_create(MPI_Comm comm_old, int ndims, int *dims, int *periods, int reorder, MPI_Comm *comm_cart);
extern(C) int MPI_Dims_create(int nnodes, int ndims, int *dims);
extern(C) int MPI_Graph_create(MPI_Comm comm_old, int nnodes, int *index, int *edges, int reorder, MPI_Comm *comm_graph);
extern(C) int MPI_Topo_test(MPI_Comm comm, int *status);
extern(C) int MPI_Graphdims_get(MPI_Comm comm, int *nnodes, int *nedges);
extern(C) int MPI_Graph_get(MPI_Comm comm, int maxindex, int maxedges, int *index, int *edges);
extern(C) int MPI_Cartdim_get(MPI_Comm comm, int *ndims);
extern(C) int MPI_Cart_get(MPI_Comm comm, int maxdims, int *dims, int *periods, int *coords);
extern(C) int MPI_Cart_rank(MPI_Comm comm, int *coords, int *rank);
extern(C) int MPI_Cart_coords(MPI_Comm comm, int rank, int maxdims, int *coords);
extern(C) int MPI_Graph_neighbors_count(MPI_Comm comm, int rank, int *nneighbors);
extern(C) int MPI_Graph_neighbors(MPI_Comm comm, int rank, int maxneighbors, int *neighbors);
extern(C) int MPI_Cart_shift(MPI_Comm comm, int direction, int disp, int *rank_source, int *rank_dest);
extern(C) int MPI_Cart_sub(MPI_Comm comm, int *remain_dims, MPI_Comm *newcomm);
extern(C) int MPI_Cart_map(MPI_Comm comm, int ndims, int *dims, int *periods, int *newrank);
extern(C) int MPI_Graph_map(MPI_Comm comm, int nnodes, int *index, int *edges, int *newrank);

/// 7.1.1.4. Clock synchronization
extern(C) int MPI_Get_processor_name(char *name, int *resultlen);

/// 7.2. Error handling
alias extern(C) void function(MPI_Comm *, int *, ...) MPI_Handler_function;

extern(C) int MPI_Errhandler_create(MPI_Handler_function *f, MPI_Errhandler *errhandler);
extern(C) int MPI_Errhandler_set(MPI_Comm comm, MPI_Errhandler errhandler);
extern(C) int MPI_Errhandler_get(MPI_Comm comm, MPI_Errhandler *errhandler);
extern(C) int MPI_Errhandler_free(MPI_Errhandler *errhandler);
extern(C) int MPI_Error_string(int errorcode, char *string, int *resultlen);
extern(C) int MPI_Error_class(int errorcode, int *errorclass);

/// 7.4. Timers and synchronization
extern(C) double MPI_Wtime();
extern(C) double MPI_Wtick();

/// 7.5. Startup
extern(C) int MPI_Init(int *argc, char ***argv);
extern(C) int MPI_Finalize();
extern(C) int MPI_Initialized(int *flag);
extern(C) int MPI_Abort(MPI_Comm comm, int errorcode);

/// 8. Profiling Interface
extern(C) int MPI_Pcontrol(const int level, ...);


class Comm {
	MPI_Comm comm_;

	int size() {
		int temp;
		MPI_Comm_size(comm_, &temp);
		return temp;
	}

	int rank() {
		int temp;
		MPI_Comm_rank(comm_, &temp);
		return temp;
	}
}

	/** Performs a basic send
	 *
	 * Note: May block.
	 */
	void Send(SendT)(Comm comm, ref SendT s, int dest, tag = MPI_ANY_TAG) {
		const SendTMPI = D_to_MPI_Datatype!(SendT);
		int ret = MPI_Send(cast(void*)&s, 1, SendTMPI, dest, tag, comm);
		if (ret == 0) {
			return;
		}
	}

/// ditto
	void Send(SendT)(Comm comm, SendT[] s, int dest, tag = MPI_ANY_TAG) {
		const SendTMPI = D_to_MPI_Datatype!(SendT);
		int ret = MPI_Send(cast(void*)s, s.length, SendTMPI, dest, tag, comm);
		if (ret == 0) {
			return;
		}
	}

	/** Performs a basic recive
	 *
	 * Note: May block.
	 */
	void Recv(SendT)(Comm comm, ref RecvT s, int source = 0, tag = MPI_ANY_TAG) {
		const SendTMPI = D_to_MPI_Datatype!(SendT);
		int ret = MPI_Recv(cast(void*)&s, 1, RecvTMPI, source, tag, comm, &status);
		if (ret == 0) {
			return;
		}
	}

/// ditto
	void Recv(RecvT)(Comm comm, RecvT[] s, int source = 0, tag = MPI_ANY_TAG) {
		const RecvTMPI = D_to_MPI_Datatype!(RecvT);
		int ret = MPI_Recv(cast(void*)s, s.length, RecvTMPI, source, tag, comm);
		if (ret == 0) {
			return;
		}
	}

/// ditto
	void Bcast(SendT)(Comm comm, ref SendT[] s, int root = 0) {
		const SendTMPI = D_to_MPI_Datatype!(SendT);
		int ret = MPI_Recv(cast(void*)s, s.length, SendTMPI, root, comm);
		if (ret == 0) {
			return;
		}
	}

/// ditto
	void Bcast(SendT)(Comm comm, ref SendT s, int root = 0)  {
		const SendTMPI = D_to_MPI_Datatype!(SendT);
		int ret = MPI_Recv(cast(void*)s, 1, SendTMPI, root, comm);
		if (ret == 0) {
			return;
		}
	}

/// ditto
	void Scatter(SendT, RecvT)(Comm comm, ref SendT[] s, int cnt, ref RecvT[] r, int root = 0) {
		const SendTMPI = D_to_MPI_Datatype!(SendT);
		const RecvTMPI = D_to_MPI_Datatype!(RecvT);
		int ret = MPI_Scatter(cast(void*)s, cnt, SendTMPI, cast(void*)r, r.length, RecvTMPI, root, comm);
		if (ret == 0) {
			return;
		}
	}

template D_to_MPI_Datatype(T) {
	static if (is(T == double)) {
		const D_to_MPI_Datatype = MPI_DOUBLE;
	} else static if (is(T == float)) {
		const D_to_MPI_Datatype = MPI_FLOAT;
	} else static if (is(T == real)) {
		static assert(real.sizeof == 10);
		const D_to_MPI_Datatype = MPI_LONG_DOUBLE;

	} else static if (is(T == int)) {
		const D_to_MPI_Datatype = MPI_INT;
	} else static if (is(T == short)) {
		const D_to_MPI_Datatype = MPI_SHORT;
	} else static if (is(T == byte)) {
		const D_to_MPI_Datatype = MPI_BYTE;

	} else static if (is(T == char)) {
		const D_to_MPI_Datatype = MPI_CHAR;
	} else static if (is(T == dchar)) {
		const D_to_MPI_Datatype = MPI_SHORT;
	} else static if (is(T == wchar)) {
		const D_to_MPI_Datatype = MPI_INT;

	} else static if (is(T == uint)) {
		const D_to_MPI_Datatype = MPI_UINT;
	} else static if (is(T == ushort)) {
		const D_to_MPI_Datatype = MPI_USHORT;
	} else static if (is(T == ubyte)) {
		const D_to_MPI_Datatype = MPI_UBYTE;

	} else static if (is(T == long)) {
		const D_to_MPI_Datatype = MPI_LONG;
	} else static if (is(T == ulong)) {
		const D_to_MPI_Datatype = MPI_ULONG;
	} else {
		static assert(0);
	}
}
